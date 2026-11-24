"""ZL Compressor 四段差异化母带密度压缩：按低/人声核心/清晰/空气分区独立控制，避免低频压制人声。"""
import threading

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from . import config as C
from .vst_worker import render_isolated

PLUGIN = C.ROOT / 'plugins/ZL Compressor.vst3/Contents/x86_64-win/ZL Compressor.vst3'
# 分频点按听感与声部边界划分：
#   Band 0 (< 220 Hz)       超低频与底鼓/贝斯：独立控峰，防止低频冲撞后级宽带限幅器压制人声
#   Band 1 (220 - 1800 Hz)  人声核心与中频：极轻柔、慢起音、大软拐点，保护人声通透与动态
#   Band 2 (1800 - 7500 Hz) 清晰存在区：温和控制尖锐峰值，不压暗人声
#   Band 3 (> 7500 Hz)      空气感与镲片：轻量润饰极高频瞬态
CROSSOVERS = (220., 1800., 7500.)
# (相对本段活动 RMS 的阈值余量 dB, 满档压缩比, 起音 ms, 释放 ms, 软拐点 dB, 频段力度权重)
BANDS = (
    (1.6, 2.45, 14.0, 95.0, 6.0, 1.00),
    (4.8, 1.55, 26.0, 130.0, 8.0, 0.55),
    (4.5, 1.75, 18.0, 110.0, 6.0, 0.70),
    (5.5, 1.85, 10.0, 80.0, 6.0, 0.65),
)
BLOCK = 2048


def _split(mixed, sr):
    rest = mixed
    for hz in CROSSOVERS:
        low = sosfiltfilt(butter(2,min(hz,sr*.45),fs=sr,output='sos'),rest,axis=-1).astype(np.float32)
        yield low
        rest = rest-low
    yield rest


def _compressor(ratio, attack, release=100., knee=6.):
    from pedalboard import load_plugin
    if not PLUGIN.is_file():
        raise RuntimeError('缺少 ZL Compressor 0.6.1，请运行 python tools/fetch_zl.py 或使用完整安装包')
    fx = load_plugin(str(PLUGIN))
    fx.compressor_style_compressor_style = 'Optical'
    fx.ratio_ratio = ratio
    fx.attack_attack = attack
    fx.release_release = release
    fx.knee_width_knee_width = knee
    fx.side_stereo_mode_side_stereo_mode = 'Left Right Max'
    fx.side_stereo_link_side_stereo_link = 100.
    return fx


def _active_band_level_db(band, sr):
    """估算单频段活动期 RMS（dBFS），避免长前奏/静音拉低参考导致过度压缩。"""
    hop = max(1, round(0.05 * sr))
    count = band.shape[-1] // hop
    if count < 2:
        power = float(np.mean(np.square(band, dtype=np.float64)))
        return 10.0 * np.log10(max(power, 1e-20))
    frames = np.mean(
        np.square(band[:, :count * hop], dtype=np.float64).reshape(2, count, hop),
        axis=(0, 2),
    )
    p95 = float(np.percentile(frames, 95))
    active = frames >= max(p95 * 0.04, 1e-10)
    mean_power = float(frames[active].mean()) if active.any() else float(frames.mean())
    return 10.0 * np.log10(max(mean_power, 1e-20))


def _tame_low_peaks(low_band, sr, band_level_db, amount):
    """仅在低频段（< 220 Hz）内吸收过突出的底鼓/贝斯峰值，防止其冲撞后级宽带限幅器压制人声。"""
    if amount <= 1e-4 or low_band.shape[-1] < sr // 4:
        return low_band
    hop = max(1, round(0.002 * sr))
    n = low_band.shape[-1]
    count = (n + hop - 1) // hop
    padded = np.pad(low_band, ((0, 0), (0, count * hop - n)))
    peaks = np.max(np.abs(padded.reshape(2, count, hop)), axis=(0, 2)).astype(np.float64)
    up, down = np.exp(-hop / (sr * 0.002)), np.exp(-hop / (sr * 0.045))
    env = np.empty(count, dtype=np.float64)
    state = 0.0
    for i, pk in enumerate(peaks):
        coeff = up if pk > state else down
        state = coeff * state + (1.0 - coeff) * pk
        env[i] = state
    env_db = 20.0 * np.log10(np.maximum(env, 1e-8))
    # 只压制超出低频平均能量 8 dB 以上的尖峰，保留正常低频弹性与饱满度
    ceiling_db = band_level_db + 8.0 + (1.0 - amount) * 3.0
    over = env_db - ceiling_db
    # 4 dB 软拐点，最大独立限峰 2.4 dB
    reduction = np.where(over < -2.0, 0.0, np.where(over > 2.0, over * 0.68, ((over + 2.0) ** 2) / 8.0 * 0.68))
    reduction = np.clip(reduction, 0.0, 2.4) * amount
    if float(np.max(reduction)) < 0.05:
        return low_band
    gain = 10.0 ** (-np.interp(np.arange(n), np.arange(count) * hop + hop / 2.0, reduction) / 20.0)
    tamed = (low_band * gain.astype(np.float32)).astype(np.float32)
    # 轻微低通清理快速包络调制产生的带外残余，确保不渗入中频人声区
    return sosfiltfilt(butter(2, min(CROSSOVERS[0] * 1.15, sr * 0.45), fs=sr, output='sos'),
                       tamed, axis=-1).astype(np.float32)


def densify(mixed, sr, input_lufs, amount, reference=None, *, checkpoint=None):
    """按频段独立基准与差异化力度执行四段母带压缩，草稿可复用整曲频段基准。"""
    amount = float(np.clip(amount,0.,1.))
    stats = {'amount':amount,'applied':False}
    if not amount or mixed.shape[-1] < sr:
        return mixed, stats
    if threading.current_thread() is not threading.main_thread():
        return render_isolated('--density-worker', mixed, sr=sr, input_lufs=input_lufs,
                               amount=amount, reference=reference, checkpoint=checkpoint)
    if reference and reference.get('applied') and len(reference.get('band_levels_db', ())) == len(BANDS):
        input_lufs = reference['input_lufs']
        band_levels = reference['band_levels_db']
    else:
        band_levels = None
    # 低密度区间平滑渐变干湿混合，消除 0 -> 1% 跳变
    wet_blend = float(np.clip(amount * 2.0, 0., 1.))
    starts = np.arange(0,mixed.shape[-1],BLOCK)
    centers = np.minimum(starts+BLOCK//2,mixed.shape[-1]-1)
    result = np.zeros_like(mixed)
    levels, reductions = [], []
    for index,(band,(headroom,ratio,attack,release,knee,weight)) in enumerate(zip(_split(mixed,sr),BANDS)):
        power = np.mean(np.square(band,dtype=np.float64),axis=0)
        level_db = _active_band_level_db(band, sr)
        ref_level = level_db if band_levels is None else band_levels[index]
        levels.append(ref_level)
        # 温和段落跟踪（收敛幅度，防止低频爆发段抬高阈值导致低频逃逸到后级限幅器）
        section = uniform_filter1d(power,round(3.*sr),mode='nearest')[centers]
        follow = 0.35 * np.clip(10*np.log10(np.maximum(section,1e-20))-ref_level, -5., 3.)
        band_amt = float(np.clip(amount * weight, 0., 1.))
        eff_ratio = 1.0 + (ratio - 1.0) * (0.30 + 0.70 * band_amt)
        base_thresh = ref_level + headroom + (1.0 - band_amt) * 4.0
        fx = _compressor(eff_ratio, attack, release=release, knee=knee)
        rendered = np.empty_like(band)
        for block,start in enumerate(starts):
            fx.threshold_db_threshold_db = float(np.clip(base_thresh+follow[block],-100.,0.))
            chunk = np.ascontiguousarray(band[:,start:start+BLOCK],dtype=np.float32)
            # 每个频段新建实例，保持分块状态并允许 Qt 后台线程处理。
            rendered[:,start:start+BLOCK] = fx.process(chunk,sr,reset=False)
        if index == 0:
            rendered = _tame_low_peaks(rendered, sr, ref_level, band_amt)
        if wet_blend < 1.0:
            rendered = band + (rendered - band) * wet_blend
        result += rendered
        hop = round(sr*.05)
        count = band.shape[-1]//hop
        original = power[:count*hop].reshape(count,hop).mean(axis=1)
        output = np.mean(np.square(rendered[:,:count*hop],dtype=np.float64)
                         .reshape(2,count,hop),axis=(0,2))
        active = original > 1e-10
        drop = 10*np.log10(original[active]/np.maximum(output[active],1e-30))
        reductions.append(round(max(0.,float(np.percentile(drop,95))),3) if drop.size else 0.)
    stats.update(applied=True,engine='ZL Compressor 0.6.1',style='Optical',
                 input_lufs=input_lufs,band_levels_db=levels,
                 band_weights=[w for *_, w in BANDS],
                 crossovers_hz=list(CROSSOVERS),band_reduction_p95_db=reductions)
    return result, stats
