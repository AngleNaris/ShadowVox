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
LOW_MID_DUCK_BAND_HZ = (150, 250)
MIX_PRESET = "single_vocal_low_mid_duck"


def bell(x, sr, frequency, gain_db, q):
    a = 10**(gain_db/40)
    w = 2*np.pi*frequency/sr
    alpha = np.sin(w)/(2*q)
    b = [1+alpha*a,-2*np.cos(w),1-alpha*a]
    den = np.array([1+alpha/a,-2*np.cos(w),1-alpha/a])
    return lfilter(np.array(b)/den[0], den/den[0], x).astype(np.float32)


def compress(x, sr, amount=1.):
    env = envelope(x,sr)
    db = 20*np.log10(np.maximum(env,1e-8))
    over = db+22
    # 6dB 软拐点，3:1，最大衰减 6dB。
    slope = 1-1/3
    reduction = np.where(over < -3,0,np.where(over>3,over*slope,(over+3)**2/12*slope))
    reduction = np.clip(reduction,0,6) * amount
    return (x*10**(-reduction/20)).astype(np.float32), float(np.max(reduction))


def deess(x, sr, amount=1.):
    """EQ 后单次温和去齿：突出高频时动态衰减，保留其他时刻的亮度。"""
    upper = min(10000, sr * .45)
    sos = butter(3,[4500,upper],btype="bandpass",fs=sr,output="sos")
    band = sosfilt(sos,x)
    high = envelope(band,sr,attack=.005,release=.08)
    total = envelope(x,sr,attack=.005,release=.08)
    relative = 20*np.log10(np.maximum(high,1e-8)/np.maximum(total,1e-8))
    active = np.clip((relative+14)/12,0,1)*np.clip((20*np.log10(np.maximum(total,1e-8))+45)/20,0,1)
    reduction = 4*active
    # 离线零相位分频，避免相位差导致频带边缘被意外提升。
    delta = sosfiltfilt(sos,x)*(1-10**(-reduction/20))
    strength = .3 * amount
    maximum = float(np.max(reduction))
    applied = -20*np.log10(1-strength*(1-10**(-maximum/20)))
    return (x-strength*delta).astype(np.float32), float(applied)


def room(x,sr,size=.66):
    """独立混响发送；延长用户选定版本的尾音，保留 20ms 前延时。"""
    source = sosfilt(butter(2,[180,min(6500,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    predelay = round(.020*sr)
    source = np.pad(source,(predelay,0))[:len(x)].astype(np.float32)
    effect = Reverb(room_size=size,damping=.45,wet_level=.102,dry_level=0,width=1)
    return effect(np.vstack([source,source]),sr)


@lru_cache(maxsize=4)
def short_room_ir(sr):
    """现场合成已选密集短房间 IR，无需外部脉冲响应文件。"""
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


def delay(x,sr):
    """独立、轻量的立体声短延时发送。"""
    source = sosfilt(butter(2,[250,min(6000,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    wet = np.zeros((2,len(x)),dtype=np.float32)
    for channel,seconds in enumerate((.110,.145)):
        for repeat,gain in ((1,.025),(2,.009)):
            offset = round(seconds*repeat*sr)
            if offset < len(x):
                wet[channel,offset:] += source[:-offset]*gain
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
    """4 倍过采样的中低频谐波；不改变共振峰。"""
    band = sosfiltfilt(butter(2,[100,min(1100,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    level = max(3*audio.rms(band), .001)
    up = resample_poly(band.astype(np.float32),4,1)/level
    harmonic = resample_poly(np.tanh(1.3*up)**2,1,4)[:len(x)]
    harmonic = sosfiltfilt(butter(3,[90,min(2600,sr*.45)],btype="bandpass",fs=sr,output="sos"),harmonic).astype(np.float32)
    target = audio.rms(x)
    harmonic *= .18*amount*target/max(audio.rms(harmonic),1e-9)
    return audio.match_rms(x+harmonic,target,max_gain_db=6).astype(np.float32)


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


def _vocal_tail(y, sr, target, eq, deessing, harmonics, smile):
    low_hz = 160. + 60.*smile
    high_hz = min(8000. - 2000.*smile, sr*.45)
    low_db = 1.2*eq + 4.8*smile
    high_db = 3.*eq + 6.*smile
    if eq:
        y = LowShelfFilter(cutoff_frequency_hz=low_hz, gain_db=low_db, q=.7071)(y,sr)
        y = HighShelfFilter(cutoff_frequency_hz=high_hz, gain_db=high_db, q=.7071)(y,sr)
    y = audio.match_rms(y,target,max_gain_db=6).astype(np.float32)
    if harmonics:
        y = warm_harmonics(y,sr,harmonics)
    y, reduction = deess(y,sr,deessing)
    return y, {"low_shelf_hz":low_hz,"low_shelf_gain_db":low_db,
               "high_shelf_hz":high_hz,"high_shelf_gain_db":high_db,"max_deess_db":reduction}


def _fill_thin_vocal(vocal, reference, sr, amount):
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
    filled = bell(vocal,sr,225,2.5,.8)
    return (vocal+(filled-vocal)*curve).astype(np.float32)


def _vocal_chain(vocal, sr, lowcut_hz, eq, compression, deessing, combined, harmonics=1.):
    smile = float(np.clip(eq-1.,0.,1.))
    eq = min(eq,1.)
    target = audio.rms(vocal)
    y = (sosfilt(butter(2,lowcut_hz,btype="highpass",fs=sr,output="sos"),vocal)
         if lowcut_hz else vocal.copy())
    if eq:
        y = bell(y,sr,350,-2.*eq,1.)
    peak_reduction, peak_threshold = 0., None
    if combined:
        y, peak_reduction, peak_threshold = peak_compress(y,sr,compression)
    y, max_reduction = compress(y,sr,compression)
    for frequency, gain in ((180,1.5),(2800,1.2)):
        if eq:
            y = bell(y,sr,frequency,gain*eq,.8)
    y = audio.match_rms(y,target,max_gain_db=6).astype(np.float32)
    for frequency, gain in ((260,1.),(500,.65)):
        if eq:
            y = bell(y,sr,frequency,gain*eq,.8)
    y = audio.match_rms(y,target,max_gain_db=6).astype(np.float32)
    harmonics = harmonics if combined else 0.
    result, stats = _vocal_tail(y,sr,target,eq,deessing,harmonics,smile)
    if smile:
        reference, _ = _vocal_tail(y,sr,target,eq,deessing,harmonics,0.)
        result = _fill_thin_vocal(result,reference,sr,smile)
    return result, {**stats,"smile_amount":smile,"dynamic_body_max_db":2.5*smile,
                    "max_compression_db":max_reduction,
                    "peak_compression_db":peak_reduction,"peak_threshold_dbfs":peak_threshold}


def _backing(accompaniment, sidechain, sr, amount):
    env = envelope(sidechain,sr,attack=.025,release=.22)
    activity = np.clip((20*np.log10(np.maximum(env,1e-8))+45)/25,0,1)
    if not amount:
        return accompaniment, activity
    band = sosfiltfilt(butter(2,[250,min(4000,sr*.45)],btype="bandpass",fs=sr,output="sos"),accompaniment,axis=-1)
    bed = accompaniment-band*(1-10**(-2.5*amount*activity/20))
    bed *= 10**(-.6*amount*activity/20)
    return bed.astype(np.float32), activity


def _low_mid_duck(bed, vocal, sr, amount):
    """人声低中频被遮住时，联动衰减伴奏的同频段，最大 2 dB。"""
    if not amount:
        return bed, 0.
    band_filter = butter(2, LOW_MID_DUCK_BAND_HZ, btype='bandpass', fs=sr, output='sos')
    bed_band = sosfiltfilt(band_filter, bed, axis=-1)
    vocal_band = sosfiltfilt(band_filter, vocal)
    hop = round(.01 * sr)
    vocal_db = 20 * np.log10(_frame_envelope(vocal_band, sr, hop, .035, .18))
    bed_db = 20 * np.log10(_frame_envelope(bed_band, sr, hop, .035, .18))
    presence = np.clip((vocal_db + 42) / 14, 0, 1)
    overlap = np.clip((bed_db - vocal_db) * amount / 6, 0, 1)
    reduction = 2 * presence * overlap
    gain = 10 ** (-_interpolate_frames(reduction, bed.shape[-1], hop) / 20)
    return (bed - bed_band * (1 - gain)).astype(np.float32), float(reduction.max())


def enhance(vocal, accompaniment, sr, space=1., lowcut_hz=75., eq=1., compression=1.,
            deessing=1., reverb=1., delay_amount=1., duck=1., reference_vocal=None,
            vocal_gain_db=0., gain_db=None, capture=None, harmonics=1.):
    """人声音色、慢速比例、空间及伴奏避让，输出母带前的立体声。"""
    y, vocal_stats = _vocal_chain(vocal,sr,lowcut_hz,eq,compression,deessing,True,harmonics)
    balance_stats = {"reference_used":False}
    if reference_vocal is not None and gain_db is None:
        # 手动人声音量不参与自动比例估计，避免自动增益抵消用户调节。
        # 遮蔽避让只参与实际混音，比例曲线用独立分析链计算。
        analysis_voice = vocal * 10**(-vocal_gain_db/20)
        analysis_y, _ = _vocal_chain(analysis_voice,sr,lowcut_hz,min(eq,1.),compression,deessing,False)
        bed, activity = _backing(accompaniment,analysis_y,sr,duck)
        baseline = (analysis_y[None,:] + early_reflections(analysis_y,sr) +
                    room(analysis_y,sr,size=.52) + delay(analysis_y,sr))
        times, curve_db, balance_stats = mix_balance.curve(analysis_y,baseline,bed,reference_vocal,accompaniment,sr,
                                                           original_dry=analysis_voice)
        gain_db = np.interp(np.arange(len(vocal))/sr,times,curve_db)
        if capture is not None:
            capture['mix_gain_curve'] = {'times':times, 'gain_db':curve_db}
    scale = (10**(np.asarray(gain_db)/20)).astype(np.float32) if gain_db is not None else 1.
    scaled_vocal = y * scale
    bed, activity = _backing(accompaniment,y,sr,duck)
    bed, low_mid_reduction = _low_mid_duck(bed, scaled_vocal, sr, duck)
    long_tail = float(np.clip(reverb-1.,0.,1.))
    short_send = min(reverb,1.)
    long_send = short_send + .4*long_tail
    room_size = .66 + .12*long_tail
    wet_duck_db = 4.5 - 1.5*long_tail
    wet = 0.
    if space:
        env = envelope(scaled_vocal,sr,attack=.025,release=.25)
        wet_activity = np.clip((20*np.log10(np.maximum(env,1e-8))+33)/18,0,1)
        wet_gain = (10**(-wet_duck_db*wet_activity/20)).astype(np.float32)
        short = 0.
        if reverb:
            short = short_room(y,sr)*short_send
            wet = room(y,sr,size=room_size)*(10**(LONG_ROOM_RETURN_DB/20)*long_send)
        if delay_amount:
            wet = wet + delay(y,sr)*delay_amount
        wet = (short + wet*wet_gain)*space
    mixed = bed + (y[None,:]+wet)*scale
    if capture is not None:
        from .report_analysis import voice_features
        capture['report_vocal_features'] = voice_features(scaled_vocal, sr)
    mid_eq = min(eq,1.)
    return mixed.astype(np.float32), {"enabled":True,"highpass_hz":lowcut_hz,
          "eq_bands":[[350,-2*mid_eq],[180,1.5*mid_eq],[2800,1.2*mid_eq],[260,mid_eq],[500,.65*mid_eq]],
          "compression_threshold_dbfs":-22,"compression_ratio":3., **vocal_stats,
          "eq_amount":eq,"compression_amount":compression,"deess_amount":deessing,"harmonics_amount":harmonics,
          "reverb_amount":reverb,"delay_amount":delay_amount,"duck_amount":duck,
          "deess_position":"after_eq","deess_strength":.3*deessing,
          "harmonics_oversampling":4,"reverb_engine":"Freeverb","reverb_room_size":room_size,
          "reverb_predelay_ms":20,"reverb_wet":.102*space*long_send,
          "reverb_return_adjustment_db":LONG_ROOM_RETURN_DB,"reverb_tail_boost_db":3.,
          "short_room_return_db":SHORT_ROOM_RETURN_DB,"short_room_predelay_ms":8,
          "mix_preset":MIX_PRESET,
          "low_mid_duck_band_hz":list(LOW_MID_DUCK_BAND_HZ),"max_low_mid_duck_db":low_mid_reduction,
          "delay_wet":.025*space*delay_amount,
          "wet_duck_max_db":wet_duck_db if space else 0.,"reference_balance":balance_stats,
          "duck_band_hz":[250,min(4000,sr*.45)],"max_band_duck_db":float(np.max(activity)*2.5*duck),
          "max_broadband_duck_db":float(np.max(activity)*.6*duck)}


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
