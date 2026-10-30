"""Slow offline vocal-to-backing level automation based on separated stems."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


def _levels(signal, indices, window):
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


def curve(dry, baseline, bed, reference, original_bed, sr, original_dry=None):
    """Return one smooth dB curve; calculate each singer's active levels independently."""
    n = len(dry)
    hop = max(1, round(.05 * sr))
    indices = np.arange(0, n, hop)
    times = indices / sr
    step = hop / sr

    dry_short = _levels(dry, indices, max(1, round(.4 * sr)))
    vocal_short = _levels(baseline, indices, max(1, round(.4 * sr)))
    bed_short = _levels(bed, indices, max(1, round(.4 * sr)))
    ratio = vocal_short - bed_short
    active = (dry_short > -29) & (bed_short > -45)
    target = np.empty(len(indices))
    if active.any():
        fallback = float(np.median(ratio[active]))
        radius = round(5 / step)
        for i in range(len(indices)):
            a, b = max(0, i-radius), min(len(indices), i+radius+1)
            local = ratio[a:b][active[a:b]]
            target[i] = np.median(local) if len(local) >= 10 else fallback
        activity = np.clip((dry_short + 35) / 6, 0, 1)
        rider = np.clip((target - ratio) * .55, -1.5, 1.5) * activity
        rider = gaussian_filter1d(rider, sigma=.6 / step, mode="nearest")
    else:
        rider = np.zeros(len(indices))

    window = max(1, round(sr))
    reference = np.pad(reference[:n], (0, max(0, n-len(reference))))
    ref = _levels(reference, indices, window)
    ref_bed = _levels(original_bed, indices, window)
    dry_db = _levels(dry if original_dry is None else original_dry, indices, window)
    rider_samples = 10 ** (np.interp(np.arange(n), indices, rider) / 20)
    vocal_db = _levels(baseline * rider_samples, indices, window)
    bed_db = _levels(bed, indices, window)
    ref_active = (ref > -32) & (ref_bed > -45)
    cover_active = (dry_db > -32) & (bed_db > -45)
    centres = np.arange(0, times[-1] + 2., 2.)
    anchors = np.full(len(centres), np.nan)
    required = round(2 / step)
    for i, centre in enumerate(centres):
        region = (times >= centre-6) & (times < centre+6)
        a, b = region & ref_active, region & cover_active
        if np.count_nonzero(a) >= required and np.count_nonzero(b) >= required:
            original_ratio = np.median((ref-ref_bed)[a])
            cover_ratio = np.median((vocal_db-bed_db)[b])
            anchors[i] = np.clip(-.5 * (cover_ratio-original_ratio), -1.5, 1.5)
    valid = np.isfinite(anchors)
    if valid.any():
        reference_gain = np.interp(times, centres[valid], anchors[valid])
        reference_gain = _limit_rate(gaussian_filter1d(reference_gain, sigma=3/step, mode="nearest"), times)
    else:
        reference_gain = np.zeros(len(times))
    combined = _limit_rate(gaussian_filter1d(reference_gain+rider, sigma=1.5/step, mode="nearest"), times)
    return times, combined, {
        "reference_used": bool(valid.any()), "reference_sections": int(valid.sum()),
        "window_seconds": 12, "step_seconds": 2, "strength": .5,
        "reference_limit_db": 1.5, "max_rate_db_per_second": .2,
        "gain_min_db": round(float(combined.min()), 3),
        "gain_max_db": round(float(combined.max()), 3),
    }
