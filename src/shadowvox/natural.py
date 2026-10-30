"""实验性音符中心修正：每段搬移一个常量，保留段内原始音高形状。"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def center_shift(dry_cents, target_cents, voiced, strength=0.85,
                 max_shift_cents=700.0):
    """100fps；参考曲线只用于分段和中心，不把输出量化成水平音符。

    90ms 中值参考用于识别半音区域，少于 100ms 的区域不单独估计修正。
    邻接长音的局部过渡仍可能影响短区域边缘。
    每段中心差沿用旧模式的保守权重；边界短过渡不跨清音。
    这是试听原型：不纠正段内慢漂移，也不保证分段对应真实乐谱音符。
    """
    shift = np.zeros(len(dry_cents), dtype=np.float64)
    for start, end in runs(voiced):
        target = target_cents[start:end]
        a4_cents = 1200 * np.log2(440 / 10)
        labels = np.rint((median_filter(target, size=9, mode="nearest") - a4_cents) / 100)
        bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 10:
                continue
            lo, hi = start + a, start + b
            delta = float(np.median(target_cents[lo:hi]) - np.median(dry_cents[lo:hi]))
            confidence = np.clip(1 - (abs(delta) - 80) / 220, 0, 1)
            shift[lo:hi] = np.clip(delta, -max_shift_cents, max_shift_cents) * confidence * strength
        shift[start:end] = gaussian_filter1d(shift[start:end], sigma=2, mode="constant")
    return shift
