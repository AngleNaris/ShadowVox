"""音区保护：乐句/持续音区决定八度，去颤音的音高中心差决定修正量。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

from .natural import runs


def fold_register(dry, reference, voiced):
    """100fps 音分。250ms 以上空隙分句；音区改变需持续至少 300ms。

    采用同一八度关系的片段内不逐帧折叠。超过 450c 的残差不自动修正，
    包括短暂八度误检、模糊音区边界及严重旋律不匹配。
    """
    octave = np.zeros(len(dry), dtype=np.int32)
    indices = np.flatnonzero(voiced)
    if not len(indices):
        return reference.copy(), np.zeros(len(dry), bool), octave
    breaks = np.r_[0, np.flatnonzero(np.diff(indices) > 25) + 1, len(indices)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        active = indices[a:b]
        start, end = active[0], active[-1] + 1
        difference = np.interp(np.arange(start, end), active, dry[active] - reference[active])
        smooth = median_filter(difference, size=21, mode="nearest")
        candidates = np.rint(smooth / 1200).astype(np.int32)
        phrase_octave = int(np.rint(np.median(difference) / 1200))
        local = np.full(end-start, phrase_octave, dtype=np.int32)
        bounds = np.r_[0, np.flatnonzero(np.diff(candidates)) + 1, len(candidates)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            k = candidates[lo]
            good = voiced[start+lo:start+hi] & (np.abs(difference[lo:hi] - 1200*k) <= 450)
            if hi-lo >= 30 and good.sum() >= .7*(hi-lo):
                local[lo:hi] = k
        octave[start:end] = local
    target = reference + 1200*octave
    reliable = voiced & (np.abs(target-dry) <= 450)
    return target, reliable, octave


def register_shift(dry, reference, voiced, strength, max_shift_cents=450):
    target, reliable, octave = fold_register(dry, reference, voiced)
    shift = np.zeros(len(dry), dtype=np.float64)
    for start, end in runs(reliable):
        if end-start < 10:
            continue
        difference = target[start:end]-dry[start:end]
        # 110ms→150ms 中值级联压掉两轨颤音的差异，保留音符跳变台阶，再 50ms 过渡：
        # 主要修音高中心，演唱者颤音/滑音大部分留在源曲线，少移植原唱抖动。
        # 代价：短于约 80ms 的音及换音时机差不单独修正，时机交给节奏模块。
        delta = median_filter(median_filter(difference, size=11, mode="nearest"),
                              size=15, mode="nearest")
        delta = uniform_filter1d(delta, size=5, mode="nearest")
        deviation = median_filter(np.abs(difference-delta), size=21, mode="nearest")
        confidence = np.clip((160-deviation)/100, 0, 1)
        shift[start:end] = np.clip(delta, -max_shift_cents, max_shift_cents)*strength*confidence
        shift[start:end] = gaussian_filter1d(shift[start:end], sigma=1, mode="constant")
    shift[~reliable] = 0
    return shift, reliable, octave
