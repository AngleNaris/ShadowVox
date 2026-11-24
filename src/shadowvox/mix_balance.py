"""Slow offline vocal-to-backing level automation based on separated stems."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def _levels(signal, indices, window, sr=None, perceptual=False):
    if perceptual and sr is not None:
        from .mixing import k_weight
        signal = k_weight(signal, sr)
    power = np.square(signal, dtype=np.float64)
    if power.ndim == 2:
        power = np.mean(power, axis=0)
    power = uniform_filter1d(power, size=window, mode="nearest")
    return 10 * np.log10(np.maximum(power[indices], 1e-16))


def _limit_rate(values, times, maximum=.2):
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        delta = maximum * (times[i] - times[i-1])
        result[i] = np.clip(values[i], result[i-1] - delta, result[i-1] + delta)
    return result


def curve(dry, baseline, bed, reference, original_bed, sr, original_dry=None,
          perceptual=False, max_boost_db=2.0):
    """Return one smooth dB curve; calculate each singer's active levels independently."""
    n = len(dry)
    hop = max(1, round(.05 * sr))
    indices = np.arange(0, n, hop)
    times = indices / sr
    step = hop / sr

    dry_short = _levels(dry, indices, max(1, round(.4 * sr)), sr, perceptual=perceptual)
    vocal_short = _levels(baseline, indices, max(1, round(.4 * sr)), sr, perceptual=perceptual)
    bed_short = _levels(bed, indices, max(1, round(.4 * sr)), sr, perceptual=perceptual)
    ratio = vocal_short - bed_short
    short_gate = -31 if perceptual else -29
    act_offset = 37 if perceptual else 35
    active = (dry_short > short_gate) & (bed_short > -45)
    target = np.empty(len(indices))
    if active.any():
        fallback = float(np.median(ratio[active]))
        radius = round(5 / step)
        for i in range(len(indices)):
            a, b = max(0, i-radius), min(len(indices), i+radius+1)
            local = ratio[a:b][active[a:b]]
            target[i] = np.median(local) if len(local) >= 10 else fallback
        activity = np.clip((dry_short + act_offset) / 6, 0, 1)
        if perceptual:
            rider = np.clip((target - ratio) * .55, -1.5, 1.5) * activity
            rider = gaussian_filter1d(rider, sigma=.5 / step, mode="nearest")
        else:
            rider = np.clip((target - ratio) * .55, -1.5, 1.5) * activity
            rider = gaussian_filter1d(rider, sigma=.6 / step, mode="nearest")
    else:
        rider = np.zeros(len(indices))

    window = max(1, round(sr))
    reference = np.pad(reference[:n], (0, max(0, n-len(reference))))
    ref = _levels(reference, indices, window, sr, perceptual=perceptual)
    ref_bed = _levels(original_bed, indices, window, sr, perceptual=perceptual)
    dry_db = _levels(dry if original_dry is None else original_dry, indices, window, sr, perceptual=perceptual)
    rider_samples = 10 ** (np.interp(np.arange(n), indices, rider) / 20)
    vocal_db = _levels(baseline * rider_samples, indices, window, sr, perceptual=perceptual)
    bed_db = _levels(bed, indices, window, sr, perceptual=perceptual)
    long_gate = -34 if perceptual else -32
    ref_active = (ref > long_gate) & (ref_bed > -45)
    cover_active = (dry_db > long_gate) & (bed_db > -45)
    centres = np.arange(0, times[-1] + 2., 2.)
    anchors = np.full(len(centres), np.nan)
    required = round(2 / step)
    strength = .72 if perceptual else .7
    for i, centre in enumerate(centres):
        region = (times >= centre-6) & (times < centre+6)
        a, b = region & ref_active, region & cover_active
        if np.count_nonzero(a) >= required and np.count_nonzero(b) >= required:
            original_ratio = np.median((ref-ref_bed)[a])
            cover_ratio = np.median((vocal_db-bed_db)[b])
            anchors[i] = np.clip(-strength * (cover_ratio-original_ratio), -3.0, max_boost_db)
    valid = np.isfinite(anchors)
    ref_rate = 0.45 if perceptual else 0.2
    combined_rate = 0.65 if perceptual else 0.2
    combined_sigma = 0.85 if perceptual else 1.5
    global_gain_db = 0.0
    orig_global = None
    cover_global = None
    target_global = None
    if perceptual and ref_active.any() and cover_active.any():
        # 1) 全局人声-伴奏感知响度比对齐：
        #    消除不同原曲混音/分离导致的极端原唱比例（如密集流行 -2.5~-3.3dB vs 稀疏钢琴 +3.6dB），
        #    收敛到一致且自然的商用人声突出区间 [0.0, +2.2] dB；并与段落动态跟踪解耦，避免顶满限幅。
        orig_global = float(np.median((ref - ref_bed)[ref_active]))
        cover_global = float(np.median((vocal_db - bed_db)[cover_active]))
        target_global = float(np.clip(0.65 * orig_global + 0.35 * 0.8, 0.0, 2.2))
        global_gain_db = float(np.clip(target_global - cover_global, -5.0, 6.0))
        # 2) 段落动态跟踪改为跟踪各自相对全局中位的起伏差（主歌/副歌对比），不重复叠加全局偏置
        section_anchors = np.full(len(centres), np.nan)
        for i, centre in enumerate(centres):
            region = (times >= centre-6) & (times < centre+6)
            a, b = region & ref_active, region & cover_active
            if np.count_nonzero(a) >= required and np.count_nonzero(b) >= required:
                orig_dev = float(np.median((ref - ref_bed)[a])) - orig_global
                cover_dev = float(np.median((vocal_db - bed_db)[b])) - cover_global
                section_anchors[i] = np.clip(-strength * (cover_dev - orig_dev), -2.5, 2.5)
        sec_valid = np.isfinite(section_anchors)
        if sec_valid.any():
            reference_gain = np.interp(times, centres[sec_valid], section_anchors[sec_valid])
            reference_gain = _limit_rate(
                gaussian_filter1d(reference_gain, sigma=2.0/step, mode="nearest"),
                times, maximum=ref_rate,
            )
            valid = sec_valid
        else:
            reference_gain = np.zeros(len(times))
        local_mod = _limit_rate(
            gaussian_filter1d(np.clip(reference_gain + rider, -2.5, 2.5), sigma=combined_sigma/step, mode="nearest"),
            times, maximum=combined_rate,
        )
        combined = global_gain_db + local_mod
    else:
        if valid.any():
            reference_gain = np.interp(times, centres[valid], anchors[valid])
            reference_gain = _limit_rate(
                gaussian_filter1d(reference_gain, sigma=(2.0 if perceptual else 3.0)/step, mode="nearest"),
                times, maximum=ref_rate,
            )
        else:
            reference_gain = np.zeros(len(times))
        raw_combined = np.clip(reference_gain + rider, -3.0, max_boost_db) if perceptual else (reference_gain + rider)
        combined = _limit_rate(
            gaussian_filter1d(raw_combined, sigma=combined_sigma/step, mode="nearest"),
            times, maximum=combined_rate,
        )
    stats = {
        "reference_used": bool(valid.any() or orig_global is not None), "reference_sections": int(valid.sum()),
        "window_seconds": 12, "step_seconds": 2, "strength": strength,
        "perceptual_weighting": bool(perceptual),
        "reference_limit_db": 3.0, "max_boost_db": float(max_boost_db),
        "max_rate_db_per_second": combined_rate,
        "gain_min_db": round(float(combined.min()), 3),
        "gain_max_db": round(float(combined.max()), 3),
    }
    if orig_global is not None:
        stats.update({
            "global_gain_db": round(global_gain_db, 3),
            "orig_global_ratio_db": round(orig_global, 2),
            "cover_global_ratio_db": round(cover_global, 2),
            "target_global_ratio_db": round(target_global, 2),
        })
    return times, combined, stats
