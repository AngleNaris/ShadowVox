"""温和人声混音：自适应频谱校正、均衡、动态、空间效果及 M/S 分频伴奏避让。"""
from copy import deepcopy
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
DIAG_BANDS = (
    ("low", 100.0, 260.0),
    ("mud", 260.0, 520.0),
    ("mid", 520.0, 1200.0),
    ("pres", 1200.0, 4000.0),
    ("sib", 4000.0, 8000.0),
    ("air", 8000.0, 14000.0),
)


@lru_cache(maxsize=8)
def _k_weight_sos(sr):
    """ITU-R BS.1770 K 计权滤波器，用于感知响度对齐。"""
    a = 10.0 ** (4.0 / 40.0)
    w = 2.0 * np.pi * 1500.0 / sr
    c, beta = np.cos(w), np.sqrt(2.0 * a) * np.sin(w)
    b = np.array([a * (a + 1 + (a - 1) * c + beta), -2 * a * (a - 1 + (a + 1) * c), a * (a + 1 + (a - 1) * c - beta)])
    d = np.array([a + 1 - (a - 1) * c + beta, 2 * (a - 1 - (a + 1) * c), a + 1 - (a - 1) * c - beta])
    shelf = np.r_[b / d[0], d / d[0]]
    w = 2.0 * np.pi * 38.0 / sr
    c, alpha = np.cos(w), np.sin(w)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    d = np.array([1 + alpha, -2 * c, 1 - alpha])
    return np.asarray([shelf, np.r_[b / d[0], d / d[0]]], dtype=np.float64)


def k_weight(x, sr):
    return sosfilt(_k_weight_sos(sr), np.asarray(x, dtype=np.float64), axis=-1).astype(np.float32)


def k_active_rms(x, sr):
    return _active_rms(k_weight(x, sr), sr)


def match_k_rms(x, target_k_rms, sr, max_gain_db=6.0):
    cur = k_active_rms(x, sr)
    if cur <= 1e-9 or target_k_rms <= 1e-9:
        return x.astype(np.float32)
    gain = float(np.clip(target_k_rms / cur, 10 ** (-max_gain_db / 20.0), 10 ** (max_gain_db / 20.0)))
    return (x * gain).astype(np.float32)


def diagnose_vocal_spectrum(x, sr, f0=None):
    """基于活动发声段分析干声 6 频段相对能量、低频重心与音高分布，识别低音男声与女声/高音。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    hop = max(1, round(0.05 * sr))
    n = len(x)
    if n < 4096:
        return None
    frames = x[: (n // hop) * hop].reshape(-1, hop)
    rms_f = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-16)
    thresh = max(float(np.percentile(rms_f, 95)) * 0.12, 10 ** (-45.0 / 20.0))
    act = rms_f >= thresh
    ref = float(np.sqrt(np.mean(rms_f[act] ** 2))) if act.any() else float(np.sqrt(np.mean(x ** 2)))
    if ref < 1e-6:
        return None
    win = np.hanning(4096)
    acc = np.zeros(2049, dtype=np.float64)
    cnt = 0
    for i in range(0, n - 4096, 2048):
        seg = x[i : i + 4096]
        if np.sqrt(np.mean(seg ** 2)) < ref * 0.15:
            continue
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        cnt += 1
    if cnt == 0:
        return None
    acc /= cnt * (win.sum() ** 2) * (ref ** 2 + 1e-16)
    hz = np.fft.rfftfreq(4096, 1.0 / sr)
    bands = {}
    for name, lo, hi in DIAG_BANDS:
        m = (hz >= lo) & (hz < min(hi, sr * 0.48))
        bands[name] = float(10.0 * np.log10(max(float(acc[m].sum()), 1e-14)))
    sub_m = (hz >= 60.0) & (hz < min(130.0, sr * 0.48))
    sub_db = float(10.0 * np.log10(max(float(acc[sub_m].sum()), 1e-14)))
    lm = (hz >= 90.0) & (hz < 500.0)
    low_power = acc[lm]
    low_centroid_hz = float(np.sum(hz[lm] * low_power) / max(float(low_power.sum()), 1e-14))
    body_total_db = float(10.0 * np.log10(10.0 ** (bands["low"] / 10.0) + 10.0 ** (bands["mud"] / 10.0)))
    f0_p10_hz, f0_median_hz = None, None
    if f0 is not None:
        vf0 = np.asarray(f0, dtype=np.float64)
        vf0 = vf0[vf0 > 40.0]
        if vf0.size >= 30:
            f0_p10_hz = round(float(np.percentile(vf0, 10)), 1)
            f0_median_hz = round(float(np.percentile(vf0, 50)), 1)
    if f0_median_hz is not None:
        is_low_male = bool(bands["low"] > -6.5 and (low_centroid_hz < 250.0 or f0_median_hz < 185.0))
        is_female_or_high = bool((not is_low_male) and f0_median_hz >= 235.0 and f0_p10_hz >= 175.0)
    else:
        is_low_male = bool(bands["low"] > -6.5 and low_centroid_hz < 250.0)
        is_female_or_high = bool(
            (not is_low_male) and (
                (low_centroid_hz >= 320.0 and bands["low"] < -9.5)
                or (low_centroid_hz >= 305.0 and bands["low"] < -11.0)
            )
        )
    return {
        "bands": {k: round(v, 2) for k, v in bands.items()},
        "sub_db": round(sub_db, 2),
        "body_total_db": round(body_total_db, 2),
        "low_centroid_hz": round(low_centroid_hz, 1),
        "pres_minus_low_db": round(bands["pres"] - bands["low"], 2),
        "sib_minus_pres_db": round(bands["sib"] - bands["pres"], 2),
        "f0_p10_hz": f0_p10_hz,
        "f0_median_hz": f0_median_hz,
        "is_low_male": is_low_male,
        "is_female_or_high": is_female_or_high,
    }


def plan_adaptive_eq(diag):
    """死区限幅的自适应频谱校正规划，区分男声与女声/高音声部并与曲风预设参数联动。"""
    b = diag["bands"]
    low, mud, pres, sib, air = b["low"], b["mud"], b["pres"], b["sib"], b["air"]
    body_total = diag["body_total_db"]
    is_female = bool(diag.get("is_female_or_high", False))
    eq_bells = []

    # 1. 低频轰鸣 / 女声基频以下冗余低频 / 声体单薄
    if is_female:
        # 女声基频主要在 200Hz 以上，100-260Hz 偏多属于近讲效应/房混轰鸣，需主动收敛
        boom_excess = max(0.0, low - (-10.2))
        boom_factor = float(np.clip(boom_excess / 3.5, 0.0, 1.0))
        if boom_excess > 0.2:
            cut_hz = float(np.clip(diag["low_centroid_hz"] * 0.65, 145.0, 210.0))
            cut_db = -float(np.clip(boom_excess * 0.95, 0.0, 3.6))
            eq_bells.append((round(cut_hz, 1), round(cut_db, 2), 0.85, "de_boom"))
        elif body_total < -8.2:
            thin_deficit = -8.2 - body_total
            boost_db = float(np.clip(thin_deficit * 0.65, 0.0, 1.8))
            eq_bells.append((220.0, round(boost_db, 2), 0.80, "body_fill"))
    else:
        # 男声/标准声部安全窗：low <= -5.8 dB 且 body_total >= -6.8 dB
        boom_excess = max(0.0, low - (-5.8))
        boom_factor = float(np.clip(boom_excess / 3.0, 0.0, 1.0))
        if boom_excess > 0.1:
            cut_hz = float(np.clip(diag["low_centroid_hz"], 155.0, 230.0))
            cut_db = -float(np.clip(boom_excess * 1.15, 0.0, 4.2))
            eq_bells.append((round(cut_hz, 1), round(cut_db, 2), 0.85, "de_boom"))
        elif body_total < -6.8:
            thin_deficit = -6.8 - body_total
            boost_db = float(np.clip(thin_deficit * 0.85, 0.0, 2.5))
            eq_bells.append((190.0, round(boost_db, 2), 0.75, "body_fill"))

    # 2. 中低频混浊（女声 260–520 Hz 更易发闷发糊，阈值更紧、削减更干脆）
    mud_limit = -5.2 if is_female else -4.5
    mud_excess = max(0.0, mud - mud_limit)
    if mud_excess > 0.15:
        if is_female:
            cut_db = -float(np.clip(mud_excess * 1.05, 0.0, 3.2))
            eq_bells.append((330.0, round(cut_db, 2), 1.05, "de_mud"))
        else:
            cut_db = -float(np.clip(mud_excess * 0.95, 0.0, 2.4))
            eq_bells.append((360.0, round(cut_db, 2), 1.1, "de_mud"))

    # 3. 清晰存在区（1.2–4 kHz，安全窗 [-13.8, -9.2] dB）
    pres_deficit = max(0.0, -13.8 - pres)
    pres_excess = max(0.0, pres - (-9.2))
    if pres_deficit > 0.2:
        g_1650 = float(np.clip(pres_deficit * 0.38, 0.0, 3.0))
        g_3200 = float(np.clip(pres_deficit * 0.54, 0.0, 4.0))
        eq_bells.append((1650.0, round(g_1650, 2), 0.75, "clarity_mid"))
        eq_bells.append((3200.0, round(g_3200, 2), 0.80, "clarity_pres"))
    elif pres_excess > 0.2:
        cut_db = -float(np.clip(pres_excess * 0.8, 0.0, 2.4))
        eq_bells.append((2800.0, round(cut_db, 2), 1.0, "tame_pres"))

    # 4. 齿音与尖锐区（4–8 kHz）：兼顾绝对能量与相对清晰区的比例
    harsh_excess = max(0.0, sib - (-14.8), (sib - pres) - (-2.5))
    harsh_factor = float(np.clip(harsh_excess / 4.5, 0.0, 1.0))
    if harsh_excess > 0.25:
        cut_db = -float(np.clip(harsh_excess * 0.68, 0.0, 3.6))
        eq_bells.append((5800.0, round(cut_db, 2), 1.15, "tame_harsh"))
    elif sib < -21.5:
        sib_deficit = -21.5 - sib
        boost_db = float(np.clip(sib_deficit * 0.48, 0.0, 2.4))
        eq_bells.append((5400.0, round(boost_db, 2), 0.85, "upper_pres"))

    # 5. 空气感（8–14 kHz，安全窗 [-24.5, -18.8] dB）
    air_shelf_delta_db = 0.0
    if air < -24.5 and harsh_excess < 1.0:
        air_deficit = -24.5 - air
        air_shelf_delta_db = float(np.clip(air_deficit * 0.36, 0.0, 2.6))
    elif air > -18.8:
        air_excess = air - (-18.8)
        air_shelf_delta_db = -float(np.clip(air_excess * 1.1, 0.0, 2.6))

    if is_female:
        # 女声默认大幅克制预设里的 160-260Hz 低架与厚实补偿，避免叠加出浑浊低频
        mud_tame = float(np.clip((mud - (-5.5)) / 2.2, 0.0, 1.0))
        female_tame = max(0.70, boom_factor, 0.90 * mud_tame)
        low_boost_scale = round(max(0.0, 1.0 - female_tame), 3)
        body_fill_scale = round(max(0.0, 1.0 - max(0.75, female_tame)), 3)
        harmonics_scale = round(max(0.35, 1.0 - 0.50 * female_tame), 3)
    else:
        low_boost_scale = round(1.0 - 0.85 * boom_factor, 3)
        body_fill_scale = round(1.0 - 1.0 * boom_factor, 3)
        harmonics_scale = round(1.0 - 0.65 * boom_factor, 3)

    return {
        "adaptive_bells": eq_bells,
        "air_shelf_delta_db": round(air_shelf_delta_db, 2),
        "low_boost_scale": low_boost_scale,
        "body_fill_scale": body_fill_scale,
        "harmonics_scale": harmonics_scale,
        "pres_boost_scale": round(1.0 - 0.7 * float(np.clip(pres_excess / 3.0, 0.0, 1.0)), 3),
        "high_shelf_scale": round(1.0 - 0.72 * harsh_factor, 3),
        "deess_scale": round(1.0 + 0.42 * harsh_factor, 3),
        "boom_factor": round(boom_factor, 3),
        "harsh_factor": round(harsh_factor, 3),
    }


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


def level_vocal(x, sr, amount=1., target_db=-18.0, max_boost_db=4.5, max_cut_db=5.5):
    """门控慢速人声电平均衡：托起轻唱句、平稳收回强音爆发，消除忽大忽小且不抬底噪。"""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-4 or len(x) < max(1, round(0.1 * sr)):
        return x.astype(np.float32), {"applied": False, "max_boost_db": 0.0, "max_cut_db": 0.0}
    hop = max(1, round(0.01 * sr))
    env = _frame_envelope(x, sr, hop, attack=0.055, release=0.32, peak=False)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-8))
    active = env_db[env_db > -40.0]
    ref_db = float(np.clip(np.median(active), -22.0, -17.0)) if active.size >= 10 else float(target_db)
    # 严格活动门限：静音与底噪（< -42 dBFS）不抬升，轻唱段（-36~-24 dBFS）顺滑托起
    up_gate = np.clip((env_db - (-42.0)) / 9.0, 0.0, 1.0)
    down_gate = np.clip((env_db - (-42.0)) / 8.0, 0.0, 1.0)
    diff = ref_db - env_db
    # 保留 ±1.5 dB 正常演唱起伏，超出后柔和拉回目标工作电平
    up = np.clip((diff - 1.5) * 0.60, 0.0, max_boost_db) * up_gate
    down = -np.clip((-diff - 1.5) * 0.68, 0.0, max_cut_db) * down_gate
    gain_db = (up + down) * amount
    gain_db = gaussian_filter1d(gain_db, sigma=max(1.0, 0.08 / (hop / sr)), mode="nearest")
    gain = 10.0 ** (_interpolate_frames(gain_db, len(x), hop) / 20.0)
    return (x * gain).astype(np.float32), {
        "applied": True,
        "target_db": round(ref_db, 2),
        "max_boost_db": round(float(np.max(gain_db)), 2),
        "max_cut_db": round(float(-np.min(gain_db)), 2),
    }


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


def warm_harmonics(x, sr, amount=1., low_hz=100.):
    """4 倍过采样的中低频谐波；按活动段 RMS 标定驱动，不受静音占比影响。"""
    lo = float(np.clip(low_hz, 90., 220.))
    band = sosfiltfilt(butter(2,[lo,min(1100,sr*.45)],btype="bandpass",fs=sr,output="sos"),x)
    active_band = _active_rms(band, sr)
    level = max(3*active_band, .001)
    up = resample_poly(band.astype(np.float32),4,1)/level
    harmonic = resample_poly(np.tanh(1.3*up)**2,1,4)[:len(x)]
    harmonic = sosfiltfilt(butter(3,[max(90., lo * 0.9),min(2600,sr*.45)],btype="bandpass",fs=sr,output="sos"),harmonic).astype(np.float32)
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


def _vocal_tail(y, sr, target, eq, deessing, harmonics, smile, profile=None, *,
                target_k=None, deess_default_max_db=6.5, harmonics_low_hz=100.):
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
    if target_k is not None:
        y = match_k_rms(y, target_k, sr, max_gain_db=6)
    else:
        y = audio.match_rms(y,target,max_gain_db=6).astype(np.float32)
    if harmonics > 1e-4:
        y = warm_harmonics(y,sr,harmonics,low_hz=harmonics_low_hz)
    # 微笑高架提升齿音频段时，同步放开去齿音补偿，避免高频发刺。
    deess_eff = deessing * (1. + .35 * smile) if deessing else 0.
    y, reduction = deess(
        y, sr, deess_eff,
        low_hz=float(profile.get('deess_low_hz', 4800.)),
        high_hz=float(profile.get('deess_high_hz', 10000.)),
        max_reduction_db=float(profile.get('deess_max_db', deess_default_max_db)),
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


def _vocal_chain(vocal, sr, lowcut_hz, eq, compression, deessing, combined, harmonics=1.,
                 profile=None, vocal_diagnosis=None, vocal_f0=None):
    profile = deepcopy(profile) if profile else {}
    adaptive_eq = bool(profile.get('adaptive_eq', False))
    smile = float(np.clip(eq-1.,0.,1.))
    eq_base = min(eq,1.)
    orig_rms = audio.rms(vocal)
    orig_active = _active_rms(vocal, sr)
    # 统一内部工作电平至 -18 dBFS 活动 RMS，使压缩/去齿不受前后曲风音量或伴奏响度影响。
    cal = (WORKING_ACTIVE_RMS / orig_active) if orig_active > 1e-6 else 1.0
    cal = float(np.clip(cal, 0.125, 8.0))
    work = (vocal * cal).astype(np.float32)
    target = audio.rms(work)
    eff_lowcut = lowcut_hz
    hp_order = 2
    y = (sosfilt(butter(hp_order,eff_lowcut,btype="highpass",fs=sr,output="sos"),work)
         if eff_lowcut else work.copy())

    diag = vocal_diagnosis
    plan = None
    target_k = None
    if adaptive_eq:
        if diag is None:
            diag = diagnose_vocal_spectrum(y, sr, f0=vocal_f0)
        if diag and eff_lowcut:
            f0_p10 = diag.get('f0_p10_hz')
            sub_db = float(diag.get('sub_db', -40.0))
            if diag.get('is_low_male') and eff_lowcut > 68.0:
                eff_lowcut = 68.0
                hp_order = 2
                y = sosfilt(butter(hp_order, eff_lowcut, btype="highpass", fs=sr, output="sos"), work)
            elif diag.get('is_female_or_high'):
                # 女声/高音基频在 180-240Hz 以上，按最低基频动态提高高通并采用 24dB/oct 彻底清理喷麦与房混低频
                target_hp = float(np.clip(f0_p10 * 0.65, 110.0, 145.0)) if f0_p10 else 120.0
                eff_lowcut = max(eff_lowcut, target_hp)
                hp_order = 4
                y = sosfilt(butter(hp_order, eff_lowcut, btype="highpass", fs=sr, output="sos"), work)
                if vocal_diagnosis is None:
                    diag = diagnose_vocal_spectrum(y, sr, f0=vocal_f0) or diag
            elif f0_p10 is not None and f0_p10 > 135.0 and sub_db > -20.0:
                # 中高男声若存在明显基频以下麦噪/低频轰鸣，也按最低基频下方安全距离提升高通
                target_hp = float(np.clip(f0_p10 * 0.62, eff_lowcut, 115.0))
                if target_hp > eff_lowcut + 2.0 or sub_db > -16.5:
                    eff_lowcut = max(eff_lowcut, target_hp)
                    hp_order = 4
                    y = sosfilt(butter(hp_order, eff_lowcut, btype="highpass", fs=sr, output="sos"), work)
                    if vocal_diagnosis is None:
                        diag = diagnose_vocal_spectrum(y, sr, f0=vocal_f0) or diag
        if diag and eq_base > 0:
            plan = plan_adaptive_eq(diag)
        target_k = k_active_rms(y, sr)
        if plan:
            for freq, gain_db, q, _ in plan['adaptive_bells']:
                y = bell(y, sr, freq, gain_db * eq_base, q)
            y = match_k_rms(y, target_k, sr, max_gain_db=6.0)
            # 联动收敛曲风固定染色，避免低频重干声或女声叠加二次增厚、尖亮干声叠加二次高架。
            if 'post_eq_bands' in profile:
                mod_post = []
                for f, g, q in profile['post_eq_bands']:
                    if f < 300.0 and g > 0:
                        g = g * plan['low_boost_scale']
                    elif 1500.0 <= f <= 4500.0 and g > 0:
                        g = g * plan['pres_boost_scale']
                    mod_post.append((f, g, q))
                profile['post_eq_bands'] = tuple(mod_post)
            else:
                default_post = ((180., 1.5, .8), (2800., 1.2, .8), (260., 1.0, .8), (500., .65, .8))
                profile['post_eq_bands'] = tuple(
                    (f, (g * plan['low_boost_scale'] if f < 300.0 and g > 0 else
                         g * plan['pres_boost_scale'] if 1500.0 <= f <= 4500.0 and g > 0 else g) * eq_base, q)
                    for f, g, q in default_post
                )
            base_low_shelf = float(profile.get('low_shelf_gain_db', 1.2 * eq_base + 2.6 * smile))
            if base_low_shelf > 0:
                profile['low_shelf_gain_db'] = base_low_shelf * plan['low_boost_scale']
            base_high_shelf = float(profile.get('high_shelf_gain_db', 3.0 * eq_base + 3.0 * smile))
            profile['high_shelf_gain_db'] = float(
                np.clip(base_high_shelf * plan['high_shelf_scale'] + plan['air_shelf_delta_db'] * eq_base,
                        -1.0, 5.2 + 1.2 * smile)
            )
            base_body = float(profile.get('dynamic_body_max_db', 2.0 * smile))
            profile['dynamic_body_max_db'] = base_body * plan['body_fill_scale']
            harmonics = harmonics * plan['harmonics_scale']
            deessing = deessing * plan['deess_scale']

    pre_bands = profile.get('pre_eq_bands', ((350., -2.0, 1.0),))
    if eq_base or 'pre_eq_bands' in profile:
        scale_eq = 1.0 if 'pre_eq_bands' in profile else eq_base
        for freq, gain_db, q in pre_bands:
            y = bell(y, sr, freq, gain_db * scale_eq, q)
    leveler_stats = {"applied": False, "max_boost_db": 0.0, "max_cut_db": 0.0}
    if adaptive_eq and combined and compression > 0:
        y, leveler_stats = level_vocal(y, sr, compression)
        if target_k is not None:
            y = match_k_rms(y, target_k, sr, max_gain_db=6.0)
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
        y = match_k_rms(y, target_k, sr, max_gain_db=6) if target_k is not None else audio.match_rms(y, target, max_gain_db=6).astype(np.float32)
        for freq, gain_db, q in post_bands[2:]:
            y = bell(y, sr, freq, gain_db * scale_eq, q)
    y = match_k_rms(y, target_k, sr, max_gain_db=6) if target_k is not None else audio.match_rms(y, target, max_gain_db=6).astype(np.float32)
    harmonics = harmonics if combined else 0.
    deess_cap = 7.5 if adaptive_eq else 6.5
    harm_low = max(100.0, float(eff_lowcut) * 1.15) if (adaptive_eq and eff_lowcut) else 100.0
    result, stats = _vocal_tail(y, sr, target, eq_base, deessing, harmonics, smile, profile=profile,
                                target_k=target_k, deess_default_max_db=deess_cap, harmonics_low_hz=harm_low)
    body_db = float(profile.get('dynamic_body_max_db', 2.0 * smile))
    if body_db > 1e-4:
        if adaptive_eq:
            result = _fill_thin_vocal(result, result, sr, 1.0, max_db=body_db)
        else:
            reference, _ = _vocal_tail(y, sr, target, eq_base, deessing, harmonics, 0., profile=profile)
            result = _fill_thin_vocal(result, reference, sr, 1.0 if 'dynamic_body_max_db' in profile else smile,
                                      max_db=body_db if 'dynamic_body_max_db' in profile else 2.0)
    # 还原至原始电平域（自适应模式保持 K 计权感知响度，避免低频增减拖累人声清晰度）
    result = (result / cal).astype(np.float32)
    if adaptive_eq:
        orig_k = k_active_rms(vocal, sr)
        if orig_k > 1e-9:
            result = match_k_rms(result, orig_k, sr, max_gain_db=6.0)
        elif orig_rms > 1e-9:
            result = audio.match_rms(result, orig_rms, max_gain_db=6).astype(np.float32)
    elif orig_rms > 1e-9:
        result = audio.match_rms(result, orig_rms, max_gain_db=6).astype(np.float32)
    eq_report = [[float(f), float(g * (1.0 if 'pre_eq_bands' in profile else eq_base))] for f, g, _ in pre_bands]
    eq_report += [[float(f), float(g * (1.0 if 'post_eq_bands' in profile else eq_base))] for f, g, _ in post_bands]
    out_stats = {**stats, "eq_bands": eq_report, "smile_amount":smile, "dynamic_body_max_db":body_db,
                 "max_compression_db":max_reduction,
                 "peak_compression_db":peak_reduction,"peak_threshold_dbfs":peak_threshold,
                 "effective_lowcut_hz":float(eff_lowcut),"highpass_order":hp_order}
    if adaptive_eq:
        out_stats["vocal_diagnosis"] = diag
        out_stats["adaptive_eq"] = plan
        out_stats["vocal_leveler"] = leveler_stats
    return result, out_stats


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


def _band_active_rms(sig, sr, lo, hi, mask):
    sos = butter(3, [lo, min(hi, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
    b = sosfiltfilt(sos, sig)
    if mask.any():
        return float(np.sqrt(np.mean(np.square(b[mask], dtype=np.float64)) + 1e-16))
    return float(np.sqrt(np.mean(np.square(b, dtype=np.float64)) + 1e-16))


def _backing_ms_duck(accompaniment, sidechain, sr, amount, diag=None):
    """声区与遮蔽感知的 M/S 双频段避让：深度让出 Mid 中置通道，保留 Side 宽度与超低频。"""
    if not amount:
        return accompaniment.astype(np.float32), {
            "enabled": False, "is_low_male": bool(diag and diag.get("is_low_male")),
            "body_band_hz": [220.0, 520.0], "pres_band_hz": [750.0, min(4200.0, sr * 0.45)],
            "mid_pres_duck_db": 0.0, "side_pres_duck_db": 0.0,
            "mid_body_duck_db": 0.0, "side_body_duck_db": 0.0,
        }
    sc_active = _active_rms(sidechain, sr)
    if sc_active <= 1e-6:
        return accompaniment.astype(np.float32), {
            "enabled": False, "is_low_male": bool(diag and diag.get("is_low_male")),
            "body_band_hz": [220.0, 520.0], "pres_band_hz": [750.0, min(4200.0, sr * 0.45)],
            "mid_pres_duck_db": 0.0, "side_pres_duck_db": 0.0,
            "mid_body_duck_db": 0.0, "side_body_duck_db": 0.0,
        }
    sc_active_db = 20.0 * np.log10(sc_active)
    env = envelope(sidechain, sr, attack=0.015, release=0.20)
    rel_db = 20.0 * np.log10(np.maximum(env, 1e-8)) - sc_active_db
    activity = np.clip((rel_db + 22.0) / 18.0, 0.0, 1.0).astype(np.float32)
    act_mask = activity > 0.35

    mid = (0.5 * (accompaniment[0] + accompaniment[1])).astype(np.float64)
    side = (0.5 * (accompaniment[0] - accompaniment[1])).astype(np.float64)

    is_low_male = bool(diag and diag.get("is_low_male"))
    body_edges = (130.0, 380.0) if is_low_male else (220.0, 520.0)
    pres_edges = (750.0, min(4200.0, sr * 0.45))

    v_pres = _band_active_rms(sidechain, sr, pres_edges[0], pres_edges[1], act_mask)
    m_pres = _band_active_rms(mid, sr, pres_edges[0], pres_edges[1], act_mask)
    pres_masking_db = 20.0 * np.log10(max(m_pres, 1e-9) / max(v_pres, 1e-9))

    v_body = _band_active_rms(sidechain, sr, body_edges[0], body_edges[1], act_mask)
    m_body = _band_active_rms(mid, sr, body_edges[0], body_edges[1], act_mask)
    body_clash_db = 20.0 * np.log10(max(m_body, 1e-9) / max(v_body, 1e-9))

    eff_amount = float(np.clip(0.55 + 0.65 * amount, 0.0, 1.35))
    pres_extra = float(np.clip((pres_masking_db + 2.0) * 0.32, 0.0, 2.2))
    mid_pres_db = float(np.clip((2.4 + pres_extra) * eff_amount, 0.0, 4.4))
    side_pres_db = mid_pres_db * 0.25

    base_body = 2.4 if is_low_male else 1.4
    body_extra = float(np.clip((body_clash_db + 6.0) * 0.20, 0.0, 1.4))
    mid_body_db = float(np.clip((base_body + body_extra) * eff_amount, 0.0, 3.4))
    side_body_db = mid_body_db * 0.18

    sos_pres = butter(3, pres_edges, btype="bandpass", fs=sr, output="sos")
    sos_body = butter(3, body_edges, btype="bandpass", fs=sr, output="sos")
    mid_pres_band = sosfiltfilt(sos_pres, mid)
    mid_body_band = sosfiltfilt(sos_body, mid)
    side_pres_band = sosfiltfilt(sos_pres, side)
    side_body_band = sosfiltfilt(sos_body, side)

    mid_out = (
        mid
        - mid_pres_band * (1.0 - 10.0 ** (-mid_pres_db * activity / 20.0))
        - mid_body_band * (1.0 - 10.0 ** (-mid_body_db * activity / 20.0))
    )
    side_out = (
        side
        - side_pres_band * (1.0 - 10.0 ** (-side_pres_db * activity / 20.0))
        - side_body_band * (1.0 - 10.0 ** (-side_body_db * activity / 20.0))
    )
    bed = np.vstack([mid_out + side_out, mid_out - side_out]).astype(np.float32)
    return bed, {
        "enabled": True,
        "is_low_male": is_low_male,
        "body_band_hz": [float(body_edges[0]), float(body_edges[1])],
        "pres_band_hz": [float(pres_edges[0]), float(pres_edges[1])],
        "pres_masking_db": round(float(pres_masking_db), 2),
        "body_clash_db": round(float(body_clash_db), 2),
        "mid_pres_duck_db": round(mid_pres_db, 2),
        "side_pres_duck_db": round(side_pres_db, 2),
        "mid_body_duck_db": round(mid_body_db, 2),
        "side_body_duck_db": round(side_body_db, 2),
    }


def _reverb_duck_gains(signal, sr, duck_db, attack_s=.010, release_s=.14, short_share=.35):
    """自适应相对门限的句尾浮现式闪避：咬字中重压湿声保清晰，乐句尾顺滑释放拉长尾音。"""
    if not len(signal) or duck_db <= 1e-4:
        ones = np.ones(len(signal), dtype=np.float32)
        return ones, ones
    active_rms = _active_rms(signal, sr)
    if active_rms <= 1e-6:
        ones = np.ones(len(signal), dtype=np.float32)
        return ones, ones
    active_db = 20. * np.log10(active_rms)
    env = envelope(signal, sr, attack=attack_s, release=release_s)
    rel_db = 20. * np.log10(np.maximum(env, 1e-8)) - active_db
    activity = np.clip((rel_db + 20.) / 15., 0., 1.)
    long_gain = (10. ** (-duck_db * activity / 20.)).astype(np.float32)
    short_gain = (10. ** (-duck_db * short_share * activity / 20.)).astype(np.float32)
    return long_gain, short_gain


def enhance(vocal, accompaniment, sr, space=1., lowcut_hz=75., eq=1., compression=1.,
            deessing=1., reverb=1., delay_amount=1., duck=1., reference_vocal=None,
            vocal_gain_db=0., gain_db=None, capture=None, harmonics=1., capture_vocal_mix=False,
            profile=None, vocal_diagnosis=None, vocal_f0=None):
    """人声音色、慢速比例、空间及伴奏避让，输出母带前的立体声。"""
    profile = profile or {}
    adaptive_eq = bool(profile.get('adaptive_eq', False))
    ms_duck = bool(profile.get('ms_duck', False))
    perceptual_balance = bool(profile.get('perceptual_balance', adaptive_eq))
    # 先撤销外部手动增益，在恒定标定域完成人声链/压缩/去齿/侧链避让，最后在推子后重新施加。
    fader = float(10 ** (vocal_gain_db / 20.0))
    base_vocal = (vocal / fader).astype(np.float32) if abs(vocal_gain_db) > 1e-6 else vocal
    y, vocal_stats = _vocal_chain(base_vocal, sr, lowcut_hz, eq, compression, deessing, True, harmonics,
                                  profile=profile, vocal_diagnosis=vocal_diagnosis, vocal_f0=vocal_f0)
    diag = vocal_stats.get('vocal_diagnosis')
    if capture is not None and diag is not None:
        capture['vocal_diagnosis'] = diag
    balance_stats = {"reference_used":False}
    if reference_vocal is not None and gain_db is None:
        if adaptive_eq:
            analysis_y = y
        else:
            analysis_y, _ = _vocal_chain(base_vocal, sr, lowcut_hz, min(eq, 1.), compression, deessing, False, profile=profile)
        if ms_duck:
            bed_bal, _ = _backing_ms_duck(accompaniment, analysis_y, sr, duck, diag=diag)
        else:
            bed_bal, _ = _backing(accompaniment, analysis_y, sr, duck)
        baseline = (analysis_y[None,:] + early_reflections(analysis_y,sr) +
                    room(analysis_y,sr,size=.52) + delay(analysis_y,sr))
        times, curve_db, balance_stats = mix_balance.curve(
            analysis_y, baseline, bed_bal, reference_vocal, accompaniment, sr,
            original_dry=base_vocal, perceptual=perceptual_balance,
            max_boost_db=2.8 if perceptual_balance else 2.0,
        )
        gain_db = np.interp(np.arange(len(vocal))/sr,times,curve_db)
        if capture is not None:
            capture['mix_gain_curve'] = {'times':times, 'gain_db':curve_db}
    scale = (10**(np.asarray(gain_db)/20)).astype(np.float32) if gain_db is not None else 1.
    post_scale = scale * fader
    scaled_vocal = y * post_scale
    ms_duck_stats = None
    if ms_duck:
        bed, ms_duck_stats = _backing_ms_duck(accompaniment, scaled_vocal, sr, duck, diag=diag)
        activity = None
    else:
        bed, activity = _backing(accompaniment, y, sr, duck)
    long_tail = float(np.clip(reverb-1.,0.,1.))
    short_send = float(profile.get('short_room_send', min(reverb, 1.) * .75))
    long_send = float(profile.get('long_room_send', min(reverb, 1.) + .35 * long_tail))
    room_size = float(profile.get('reverb_room_size', .66 + .12 * long_tail))
    room_damping = float(profile.get('reverb_damping', .45))
    room_predelay = float(profile.get('reverb_predelay_ms', 22.))
    room_lowcut = float(profile.get('reverb_lowcut_hz', 180.))
    if diag and (diag.get('is_low_male') or diag.get('is_female_or_high')):
        room_lowcut = max(room_lowcut, 260.)
    room_highcut = float(profile.get('reverb_highcut_hz', 6500.))
    wet_duck_db = float(profile.get('wet_duck_db', 7.5 - .8 * long_tail))
    wet_duck_attack = float(profile.get('wet_duck_attack_ms', 10.)) / 1000.
    wet_duck_release = float(profile.get('wet_duck_release_ms', 140.)) / 1000.
    short_duck_share = float(profile.get('short_duck_share', .35))
    delay_times = tuple(profile.get('delay_times_ms', (110., 145.)))
    delay_gains = tuple(profile.get('delay_gains', (.025, .009)))
    delay_highcut = float(profile.get('delay_highcut_hz', 6000.))
    wet = 0.
    if space:
        wet_gain, short_gain = _reverb_duck_gains(
            y * scale, sr, wet_duck_db,
            attack_s=wet_duck_attack, release_s=wet_duck_release, short_share=short_duck_share,
        )
        short = 0.
        if reverb:
            short = short_room(y, sr) * short_send * short_gain
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
    duck_report = (
        {
            "duck_band_hz": ms_duck_stats["pres_band_hz"],
            "duck_body_band_hz": ms_duck_stats["body_band_hz"],
            "max_band_duck_db": ms_duck_stats["mid_pres_duck_db"],
            "max_broadband_duck_db": 0.,
            "max_low_mid_duck_db": ms_duck_stats["mid_body_duck_db"],
            "ms_duck": ms_duck_stats,
        }
        if ms_duck_stats is not None else
        {
            "duck_band_hz": [MID_DUCK_BAND_HZ[0], min(MID_DUCK_BAND_HZ[1], sr*.45)],
            "max_band_duck_db": float(np.max(activity)*MID_DUCK_MAX_DB*duck),
            "max_broadband_duck_db": 0.,
            "max_low_mid_duck_db": 0.,
        }
    )
    return mixed.astype(np.float32), {"enabled":True,"highpass_hz":vocal_stats["effective_lowcut_hz"],
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
          **duck_report}


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
