"""温和人声混音：均衡、动态、空间效果及伴奏避让。"""
import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import HighShelfFilter, LowShelfFilter, Reverb
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter, resample_poly, oaconvolve

from . import audio, mix_balance

envelope = audio.envelope
# 已选短房间 C、长尾 +3 dB 的固定返回量，不随每次渲染重新匹配湿声 RMS。
SHORT_ROOM_RETURN_DB = -15.438630706242808
LONG_ROOM_RETURN_DB = -1.3343912329744447 + 3.
MID_DUCK_BAND_HZ = (250, 4000)
MID_DUCK_MAX_DB = 2.5 * .85
MIX_PRESET = "single_vocal_midrange_duck"
WORKING_ACTIVE_RMS = 10 ** (-18.0 / 20.0)


def _active_rms(x, sr, floor_db=-45.0):
    """只统计活动发声段的 RMS，避免长前奏/间奏静音拉低电平估计。"""
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return 0.0
    hop = max(1, round(.02 * sr))
    count = len(x) // hop
    if count < 2:
        return audio.rms(x)
    frames = x[:count * hop].reshape(count, hop)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    floor = 10 ** (floor_db / 20.0)
    peak_rms = float(np.percentile(frame_rms, 95))
    thresh = max(floor, peak_rms * 0.12)
    active = frame_rms >= thresh
    if not active.any():
        return audio.rms(x)
    return float(np.sqrt(np.mean(np.square(frame_rms[active]))))


def bell(x, sr, frequency, gain_db, q):
    if abs(gain_db) < 1e-4 or frequency <= 0 or frequency >= sr * .48:
        return x.astype(np.float32)
    a = 10**(gain_db/40)
    w = 2*np.pi*frequency/sr
    alpha = np.sin(w)/(2*q)
    b = [1+alpha*a,-2*np.cos(w),1-alpha*a]
    den = np.array([1+alpha/a,-2*np.cos(w),1-alpha/a])
    return lfilter(np.array(b)/den[0], den/den[0], x).astype(np.float32)


def compress(x, sr, amount=1., threshold_db=-22., ratio=3., max_reduction_db=6.):
    env = envelope(x,sr)
    db = 20*np.log10(np.maximum(env,1e-8))
    over = db - threshold_db
    # 6dB 软拐点
    slope = 1. - 1. / max(1.01, float(ratio))
    reduction = np.where(over < -3,0,np.where(over>3,over*slope,(over+3)**2/12*slope))
    reduction = np.clip(reduction,0,max_reduction_db) * amount
    return (x*10**(-reduction/20)).astype(np.float32), float(np.max(reduction))


def deess(x, sr, amount=1., low_hz=4800., high_hz=10000., max_reduction_db=6.5, strength_scale=.65):
    """EQ 后单次零相位动态去齿：检测与衰减支路时间严格对齐，避免刺耳齿音。"""
    if not amount:
        return x.astype(np.float32), 0.
    upper = min(high_hz, sr * .45)
    lower = min(low_hz, upper * .8)
    sos = butter(3,[lower,upper],btype="bandpass",fs=sr,output="sos")
    # 检测与相减统一采用零相位滤波，避免群延迟导致的包络错位。
    band = sosfiltfilt(sos,x)
    high = envelope(band,sr,attack=.004,release=.07)
    total = envelope(x,sr,attack=.004,release=.07)
    relative = 20*np.log10(np.maximum(high,1e-8)/np.maximum(total,1e-8))
    active = np.clip((relative+15)/12,0,1)*np.clip((20*np.log10(np.maximum(total,1e-8))+45)/20,0,1)
    reduction = max_reduction_db * active
    delta = band * (1-10**(-reduction/20))
    strength = float(np.clip(strength_scale * amount, 0., 1.))
    maximum = float(np.max(reduction))
    applied = -20*np.log10(max(1e-6, 1-strength*(1-10**(-maximum/20))))
    return (x-strength*delta).astype(np.float32), float(applied)


def room(x, sr, size=.66, damping=.45, predelay_ms=20., lowcut_hz=180., highcut_hz=6500.):
    """恢复旧版已试听确认的 Freeverb 混响发送，保留前延时与可定制参数。"""
    if not len(x):
        return np.zeros((2, 0), dtype=np.float32)
    hi = min(highcut_hz, sr * .45)
    lo = min(lowcut_hz, hi * .5)
    source = sosfilt(butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), x)
    predelay = max(0, round(predelay_ms / 1000. * sr))
    source = np.pad(source, (predelay, 0))[:len(x)].astype(np.float32)
    effect = Reverb(room_size=float(np.clip(size, 0., 1.)), damping=float(np.clip(damping, 0., 1.)),
                    wet_level=.102, dry_level=0., width=1.)
    return effect(np.vstack([source, source]), sr)


@lru_cache(maxsize=4)
def short_room_ir(sr):
    """恢复旧版已试听确认的 Freeverb 密集短房间 IR。"""
    impulse = np.zeros((2, 3*sr), dtype=np.float32)
    impulse[:, 0] = 1.
    raw = Reverb(room_size=.08, damping=.6, wet_level=1., dry_level=0., width=.7)(impulse, sr)
    ir = resample_poly(raw, 3, 4, axis=1)
    first = np.flatnonzero(np.max(np.abs(ir), axis=0) > 1e-8)[0]
    ir = ir[:, max(0, first-round(.008*sr)):]
    return sosfilt(butter(2, [180, min(6500, sr*.45)], btype='bandpass',
                         fs=sr, output='sos'), ir, axis=1).astype(np.float32)


def short_room(x, sr):
    ir = short_room_ir(sr)
    return np.stack([oaconvolve(x, channel, mode='full')[:len(x)] for channel in ir]
                    ).astype(np.float32)*10**(SHORT_ROOM_RETURN_DB/20)


def delay(x, sr, times_ms=(110., 145.), gains=(.025, .009), lowcut_hz=250., highcut_hz=6000.):
    """独立、轻量的立体声延时发送，支持曲风自定义时间与带宽。"""
    hi = min(highcut_hz, sr * .45)
    lo = min(lowcut_hz, hi * .5)
    source = sosfilt(butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), x)
    wet = np.zeros((2, len(x)), dtype=np.float32)
    for channel, ms in enumerate(times_ms):
        seconds = ms / 1000.0
        for repeat, gain in enumerate(gains, 1):
            offset = round(seconds * repeat * sr)
            if 0 < offset < len(x):
                wet[channel, offset:] += source[:-offset] * gain
    return wet


def _frame_envelope(signal, sr, hop, attack, release, peak=False):
    if signal.ndim == 2:
        signal = np.sqrt(np.mean(np.square(signal, dtype=np.float64), axis=0))
    count = (len(signal) + hop - 1) // hop
    frames = np.pad(signal, (0, count*hop-len(signal))).reshape(count, hop)
    levels = (np.max(np.abs(frames), axis=1) if peak else
              np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-16))
    result = np.empty(count)
    up, down = np.exp(-hop/(sr*attack)), np.exp(-hop/(sr*release))
    state = 0.
    for i, level in enumerate(levels):
        coefficient = up if level > state else down
        state = coefficient*state + (1-coefficient)*level
        result[i] = state
    return result


def _interpolate_frames(values, length, hop):
    return np.interp(np.arange(length), np.arange(len(values))*hop+hop/2, values)


def peak_compress(x, sr, amount=1.):
    """快速峰值压缩，阈值按当前人声的活动段估计。"""
    hop = max(1, round(.002*sr))
    env = _frame_envelope(x, sr, hop, .003, .060, peak=True)
    db = 20*np.log10(np.maximum(env,1e-8))
    active = db[db > -40]
    if not len(active) or not amount:
        return x.astype(np.float32), 0., None
    threshold = float(np.median(active))
    reduction = np.clip((db-threshold)*.75, 0, 3.5)*amount
    gain = 10**(-_interpolate_frames(reduction,len(x),hop)/20)
    return (x*gain).astype(np.float32), float(np.max(reduction)), threshold


def warm_harmonics(x, sr, amount=1.):
    """4 倍过采样的中低频谐波；按活动段 RMS 标定驱动，不受静音占比影响。"""
    band = sosfiltfilt(butter(2,[100,min(1100,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    active_band = _active_rms(band, sr)
    level = max(3*active_band, .001)
    up = resample_poly(band.astype(np.float32),4,1)/level
    harmonic = resample_poly(np.tanh(1.3*up)**2,1,4)[:len(x)]
    harmonic = sosfiltfilt(butter(3,[90,min(2600,sr*.45)],btype="bandpass",fs=sr,output="sos"),harmonic).astype(np.float32)
    target = _active_rms(x, sr)
    harm_rms = _active_rms(harmonic, sr)
    if target > 1e-9 and harm_rms > 1e-9:
        harmonic *= .16*amount*target/harm_rms
    return audio.match_rms(x+harmonic, audio.rms(x), max_gain_db=6).astype(np.float32)


def early_reflections(x, sr):
    source = sosfilt(butter(2,[180,min(6500,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    out = np.zeros((2,len(x)),dtype=np.float32)
    for channel, taps in enumerate((((.018,.042),(.025,.025),(.032,.015)),
                                    ((.016,.040),(.028,.028),(.035,.015)))):
        for seconds, gain in taps:
            offset = round(seconds*sr)
            if offset < len(x):
                out[channel,offset:] += source[:-offset]*gain
    return out


def _vocal_tail(y, sr, target, eq, deessing, harmonics, smile, profile=None):
    profile = profile or {}
    low_hz = float(profile.get('low_shelf_hz', 160. + 40. * smile))
    high_hz = float(min(profile.get('high_shelf_hz', 8000. - 1000. * smile), sr * .45))
    low_db = float(profile.get('low_shelf_gain_db', 1.2 * eq + 2.6 * smile))
    high_db = float(profile.get('high_shelf_gain_db', 3.0 * eq + 3.0 * smile))
    if eq or ('low_shelf_gain_db' in profile or 'high_shelf_gain_db' in profile):
        if abs(low_db) > 1e-4:
            y = LowShelfFilter(cutoff_frequency_hz=low_hz, gain_db=low_db, q=.7071)(y,sr)
        if abs(high_db) > 1e-4:
            y = HighShelfFilter(cutoff_frequency_hz=high_hz, gain_db=high_db, q=.7071)(y,sr)
    y = audio.match_rms(y,target,max_gain_db=6).astype(np.float32)
    if harmonics:
        y = warm_harmonics(y,sr,harmonics)
    # 微笑高架提升齿音频段时，同步放开去齿音补偿，避免高频发刺。
    deess_eff = deessing * (1. + .35 * smile) if deessing else 0.
    y, reduction = deess(
        y, sr, deess_eff,
        low_hz=float(profile.get('deess_low_hz', 4800.)),
        high_hz=float(profile.get('deess_high_hz', 10000.)),
        max_reduction_db=float(profile.get('deess_max_db', 6.5)),
    )
    return y, {"low_shelf_hz":low_hz,"low_shelf_gain_db":low_db,
               "high_shelf_hz":high_hz,"high_shelf_gain_db":high_db,"max_deess_db":reduction}


def _fill_thin_vocal(vocal, reference, sr, amount, max_db=2.0):
    """用标准均衡的人声估计偏薄时刻，再对同一条人声补声体。"""
    band = sosfiltfilt(butter(2,[150,350],btype="bandpass",fs=sr,output="sos"),reference)
    hop = round(sr*.01)
    level = 20*np.log10(_frame_envelope(reference,sr,hop,.065,.145))
    contrast = 20*np.log10(_frame_envelope(band,sr,hop,.065,.145))-level
    active = level > max(np.percentile(level,95)-16,-43)
    if not active.any():
        return vocal
    target = np.percentile(contrast[active],60)
    gain = np.clip((target-contrast)/6,0,1)*np.clip((level+43)/10,0,1)
    curve = amount*_interpolate_frames(gaussian_filter1d(gain,4),len(vocal),hop)
    filled = bell(vocal,sr,225,max_db,.8)
    return (vocal+(filled-vocal)*curve).astype(np.float32)


def _vocal_chain(vocal, sr, lowcut_hz, eq, compression, deessing, combined, harmonics=1., profile=None):
    profile = profile or {}
    smile = float(np.clip(eq-1.,0.,1.))
    eq_base = min(eq,1.)
    orig_rms = audio.rms(vocal)
    orig_active = _active_rms(vocal, sr)
    # 统一内部工作电平至 -18 dBFS 活动 RMS，使压缩/去齿不受前后曲风音量或伴奏响度影响。
    cal = (WORKING_ACTIVE_RMS / orig_active) if orig_active > 1e-6 else 1.0
    cal = float(np.clip(cal, 0.125, 8.0))
    work = (vocal * cal).astype(np.float32)
    target = audio.rms(work)
    y = (sosfilt(butter(2,lowcut_hz,btype="highpass",fs=sr,output="sos"),work)
         if lowcut_hz else work.copy())
    pre_bands = profile.get('pre_eq_bands', ((350., -2.0, 1.0),))
    if eq_base or 'pre_eq_bands' in profile:
        scale_eq = 1.0 if 'pre_eq_bands' in profile else eq_base
        for freq, gain_db, q in pre_bands:
            y = bell(y, sr, freq, gain_db * scale_eq, q)
    peak_reduction, peak_threshold = 0., None
    if combined:
        y, peak_reduction, peak_threshold = peak_compress(y,sr,compression)
    y, max_reduction = compress(
        y, sr, compression,
        threshold_db=float(profile.get('comp_threshold_db', -22.)),
        ratio=float(profile.get('comp_ratio', 3.)),
        max_reduction_db=float(profile.get('comp_max_reduction_db', 6.)),
    )
    post_bands = profile.get('post_eq_bands', ((180., 1.5, .8), (2800., 1.2, .8), (260., 1.0, .8), (500., .65, .8)))
    if eq_base or 'post_eq_bands' in profile:
        scale_eq = 1.0 if 'post_eq_bands' in profile else eq_base
        for freq, gain_db, q in post_bands[:2]:
            y = bell(y, sr, freq, gain_db * scale_eq, q)
        y = audio.match_rms(y, target, max_gain_db=6).astype(np.float32)
        for freq, gain_db, q in post_bands[2:]:
            y = bell(y, sr, freq, gain_db * scale_eq, q)
    y = audio.match_rms(y, target, max_gain_db=6).astype(np.float32)
    harmonics = harmonics if combined else 0.
    result, stats = _vocal_tail(y, sr, target, eq_base, deessing, harmonics, smile, profile=profile)
    body_db = float(profile.get('dynamic_body_max_db', 2.0 * smile))
    if body_db > 1e-4:
        reference, _ = _vocal_tail(y, sr, target, eq_base, deessing, harmonics, 0., profile=profile)
        result = _fill_thin_vocal(result, reference, sr, 1.0 if 'dynamic_body_max_db' in profile else smile,
                                  max_db=body_db if 'dynamic_body_max_db' in profile else 2.0)
    # 还原至原始电平域
    result = (result / cal).astype(np.float32)
    if orig_rms > 1e-9:
        result = audio.match_rms(result, orig_rms, max_gain_db=6).astype(np.float32)
    eq_report = [[float(f), float(g * (1.0 if 'pre_eq_bands' in profile else eq_base))] for f, g, _ in pre_bands]
    eq_report += [[float(f), float(g * (1.0 if 'post_eq_bands' in profile else eq_base))] for f, g, _ in post_bands]
    return result, {**stats, "eq_bands": eq_report, "smile_amount":smile, "dynamic_body_max_db":body_db,
                    "max_compression_db":max_reduction,
                    "peak_compression_db":peak_reduction,"peak_threshold_dbfs":peak_threshold}


def _backing(accompaniment, sidechain, sr, amount):
    env = envelope(sidechain,sr,attack=.025,release=.22)
    activity = np.clip((20*np.log10(np.maximum(env,1e-8))+45)/25,0,1)
    if not amount:
        return accompaniment, activity
    edges = (MID_DUCK_BAND_HZ[0], min(MID_DUCK_BAND_HZ[1], sr*.45))
    # Preserve the original band shape and narrow its skirts without increasing cut near the edges.
    bandpass = np.vstack([butter(order, edges, btype="bandpass", fs=sr, output="sos") for order in (2, 6)])
    band = sosfiltfilt(bandpass, accompaniment, axis=-1)
    bed = accompaniment-band*(1-10**(-MID_DUCK_MAX_DB*amount*activity/20))
    return bed.astype(np.float32), activity


def enhance(vocal, accompaniment, sr, space=1., lowcut_hz=75., eq=1., compression=1.,
            deessing=1., reverb=1., delay_amount=1., duck=1., reference_vocal=None,
            vocal_gain_db=0., gain_db=None, capture=None, harmonics=1., capture_vocal_mix=False,
            profile=None):
    """人声音色、慢速比例、空间及伴奏避让，输出母带前的立体声。"""
    profile = profile or {}
    # 先撤销外部手动增益，在恒定标定域完成人声链/压缩/去齿/侧链避让，最后在推子后重新施加。
    fader = float(10 ** (vocal_gain_db / 20.0))
    base_vocal = (vocal / fader).astype(np.float32) if abs(vocal_gain_db) > 1e-6 else vocal
    y, vocal_stats = _vocal_chain(base_vocal, sr, lowcut_hz, eq, compression, deessing, True, harmonics, profile=profile)
    balance_stats = {"reference_used":False}
    if reference_vocal is not None and gain_db is None:
        analysis_y, _ = _vocal_chain(base_vocal, sr, lowcut_hz, min(eq, 1.), compression, deessing, False, profile=profile)
        bed, _ = _backing(accompaniment, analysis_y, sr, duck)
        baseline = (analysis_y[None,:] + early_reflections(analysis_y,sr) +
                    room(analysis_y,sr,size=.52) + delay(analysis_y,sr))
        times, curve_db, balance_stats = mix_balance.curve(analysis_y,baseline,bed,reference_vocal,accompaniment,sr,
                                                           original_dry=base_vocal)
        gain_db = np.interp(np.arange(len(vocal))/sr,times,curve_db)
        if capture is not None:
            capture['mix_gain_curve'] = {'times':times, 'gain_db':curve_db}
    scale = (10**(np.asarray(gain_db)/20)).astype(np.float32) if gain_db is not None else 1.
    post_scale = scale * fader
    scaled_vocal = y * post_scale
    bed, activity = _backing(accompaniment, y, sr, duck)
    long_tail = float(np.clip(reverb-1.,0.,1.))
    short_send = float(profile.get('short_room_send', min(reverb, 1.)))
    long_send = float(profile.get('long_room_send', min(reverb, 1.) + .4 * long_tail))
    room_size = float(profile.get('reverb_room_size', .66 + .12 * long_tail))
    room_damping = float(profile.get('reverb_damping', .45))
    room_predelay = float(profile.get('reverb_predelay_ms', 20.))
    room_lowcut = float(profile.get('reverb_lowcut_hz', 180.))
    room_highcut = float(profile.get('reverb_highcut_hz', 6500.))
    wet_duck_db = float(profile.get('wet_duck_db', 3.8 - 1.5 * long_tail))
    delay_times = tuple(profile.get('delay_times_ms', (110., 145.)))
    delay_gains = tuple(profile.get('delay_gains', (.025, .009)))
    delay_highcut = float(profile.get('delay_highcut_hz', 6000.))
    wet = 0.
    if space:
        env = envelope(y * scale, sr, attack=.025, release=.25)
        wet_activity = np.clip((20*np.log10(np.maximum(env,1e-8))+33)/18,0,1)
        wet_gain = (10**(-wet_duck_db*wet_activity/20)).astype(np.float32)
        short = 0.
        if reverb:
            short = short_room(y, sr) * short_send
            wet = room(y, sr, size=room_size, damping=room_damping, predelay_ms=room_predelay,
                       lowcut_hz=room_lowcut, highcut_hz=room_highcut) * (10**(LONG_ROOM_RETURN_DB/20)*long_send)
        if delay_amount:
            wet = wet + delay(y, sr, times_ms=delay_times, gains=delay_gains, highcut_hz=delay_highcut) * delay_amount
        wet = (short + wet * wet_gain) * space
    vocal_mix = (y[None,:] + wet) * post_scale
    mixed = bed + vocal_mix
    if capture is not None:
        from .report_analysis import voice_features
        capture['report_vocal_features'] = voice_features(scaled_vocal, sr)
        if capture_vocal_mix:
            capture['vocal_mix'] = np.broadcast_to(vocal_mix, mixed.shape)
    return mixed.astype(np.float32), {"enabled":True,"highpass_hz":lowcut_hz,
          "compression_threshold_dbfs":float(profile.get('comp_threshold_db', -22)),
          "compression_ratio":float(profile.get('comp_ratio', 3.)), **vocal_stats,
          "eq_amount":eq,"compression_amount":compression,"deess_amount":deessing,"harmonics_amount":harmonics,
          "reverb_amount":reverb,"delay_amount":delay_amount,"duck_amount":duck,
          "deess_position":"after_eq","deess_strength":.65*deessing,
          "harmonics_oversampling":4,"reverb_engine":"Freeverb","reverb_room_size":room_size,
          "reverb_predelay_ms":room_predelay,"reverb_wet":.102*space*long_send,
          "reverb_return_adjustment_db":LONG_ROOM_RETURN_DB,"reverb_tail_boost_db":3.,
          "short_room_return_db":SHORT_ROOM_RETURN_DB,"short_room_predelay_ms":8,
          "mix_preset":profile.get('name', MIX_PRESET),
          "delay_wet":delay_gains[0]*space*delay_amount,
          "wet_duck_max_db":wet_duck_db if space else 0.,"reference_balance":balance_stats,
          "duck_band_hz":[MID_DUCK_BAND_HZ[0],min(MID_DUCK_BAND_HZ[1],sr*.45)],
          "max_band_duck_db":float(np.max(activity)*MID_DUCK_MAX_DB*duck),
          "max_broadband_duck_db":0.,"max_low_mid_duck_db":0.}


def measure_loudness(path):
    p = subprocess.run([audio._ffmpeg_exe(),"-v","info","-nostats","-i",str(path),
                        "-af","loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json","-f","null","-"],
                       capture_output=True,check=True,**audio.subprocess_options())
    log = p.stderr.decode(errors="replace")
    data = json.loads(log[log.rfind('{'):log.rfind('}')+1])
    return float(data["input_i"]),float(data["input_tp"])


def bus_compress(mixed, sr):
    """双声道联动轻压缩；输入先按整曲响度校准至约 -19 LUFS。"""
    hop = max(1,round(.010*sr))
    env = _frame_envelope(mixed,sr,hop,.03,.18)
    db = 20*np.log10(np.maximum(env,1e-8))
    reduction = np.clip((db+21)*.5,0,1.8)
    gain = (10**(-_interpolate_frames(reduction,mixed.shape[1],hop)/20)).astype(np.float32)
    return mixed*gain, {"enabled":True,"threshold_dbfs":-21,"ratio":2.,
                        "attack_ms":30,"release_ms":180,"max_reduction_db":float(np.max(reduction)),
                        "median_reduction_db":float(np.median(reduction))}


def master(mixed,sr,target_lufs=-16.,true_peak_ceiling=-1.5,apply_glue=False,
           density=None,density_reference=None):
    with tempfile.TemporaryDirectory(prefix="shadowvox-master-") as folder:
        path = Path(folder)/"mix.wav"
        sf.write(path,mixed.T,sr,subtype="FLOAT")
        input_lufs, peak = measure_loudness(path)
        if not np.isfinite(input_lufs) or not np.isfinite(peak):
            return mixed.astype(np.float32),{"input_lufs":None,"master_gain_db":0.,
                    "target_lufs":target_lufs,"true_peak_ceiling_dbtp":true_peak_ceiling}
        pre_gain = 0.
        bus_stats = {"enabled":False}
        density_stats = {"amount":density,"applied":False}
        lufs = input_lufs
        if density:
            from .density import densify
            mixed, density_stats = densify(mixed,sr,input_lufs,density,density_reference)
            sf.write(path,mixed.T,sr,subtype="FLOAT")
            lufs, peak = measure_loudness(path)
        elif density is None and apply_glue:
            pre_gain = min(-19.-input_lufs,12.)
            mixed, bus_stats = bus_compress(mixed*10**(pre_gain/20),sr)
            sf.write(path,mixed.T,sr,subtype="FLOAT")
            lufs, peak = measure_loudness(path)
        # 限制器最多承担约 3 dB 的峰值处理，避免为响度过度压缩。
        gain = min(target_lufs-lufs,true_peak_ceiling-peak+3.,12.-pre_gain)
        result = (mixed*10**(gain/20)).astype(np.float32)
        limiter = peak+gain > true_peak_ceiling
        if limiter:
            sf.write(path,result.T,sr,subtype="FLOAT")
            limited = Path(folder)/"limited.wav"
            limit = .84 * 10**((true_peak_ceiling+1.5)/20)
            filt = f"aresample={sr*4},alimiter=limit={limit}:attack=5:release=60:level=0:latency=1,aresample={sr}"
            subprocess.run([audio._ffmpeg_exe(),"-v","error","-y","-i",str(path),
                            "-af",filt,"-c:a","pcm_f32le",str(limited)],
                           capture_output=True,check=True,**audio.subprocess_options())
            result = sf.read(limited,dtype="float32",always_2d=True)[0].T
            result = np.pad(result,((0,0),(0,max(0,mixed.shape[1]-result.shape[1]))))[:,:mixed.shape[1]]
        sf.write(path,result.T,sr,subtype="FLOAT")
        output_lufs, output_peak = measure_loudness(path)
        correction = min(0.,true_peak_ceiling-.05-output_peak) if np.isfinite(output_peak) else 0.
        result *= 10**(correction/20)
    return result,{"input_lufs":input_lufs,"master_gain_db":pre_gain+gain,"limiter_used":limiter,
                   "bus_normalization_db":pre_gain,"bus_compression":bus_stats,"density":density_stats,
                   "true_peak_correction_db":correction,
                   "estimated_output_lufs":output_lufs+correction if np.isfinite(output_lufs) else None,
                   "measured_output_true_peak_dbtp":output_peak+correction if np.isfinite(output_peak) else None,
                   "target_lufs":target_lufs,"true_peak_ceiling_dbtp":true_peak_ceiling}
