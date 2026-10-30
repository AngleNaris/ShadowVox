"""TD-PSOLA 渲染（Praat / parselmouth）：直接在原始波形上按音分曲线移调。

按 Praat 分析的音高点做相对移调，未修音区域逐采样旁路。
修音区域仍经过周期拼接；质量依赖周期标注，气声或误检处可能出现瑕疵。
"""
import numpy as np
import parselmouth
from parselmouth.praat import call

from .natural import runs


def _controls(shift_cents, sr, length, frame_rate):
    """修音区内部用 50ms 平滑启停；不向零移调/保护区扩张。"""
    shift = np.asarray(shift_cents, dtype=np.float64).copy()
    blend = np.zeros(length, dtype=np.float32)

    def envelope(times, start, end):
        phase = np.clip(np.minimum(times-start, end-times) / .05, 0., 1.)
        return phase*phase*(3.-2.*phase)

    for lo, hi in runs(np.abs(shift_cents) > 1e-3):
        start, end = lo/frame_rate, min(hi/frame_rate, length/sr)
        shift[lo:hi] *= envelope(np.arange(lo, hi)/frame_rate, start, end)
        if not np.any(np.abs(shift[lo:hi]) > 1e-3):
            shift[lo:hi] = 0.
            continue
        first, last = min(length, int(np.ceil(start*sr))), min(length, int(np.ceil(end*sr)))
        blend[first:last] = envelope(np.arange(first, last)/sr, start, end)
    shift[np.abs(shift_cents) <= 1e-3] = 0.
    return shift, blend


def render(audio, sr, shift_cents, frame_rate=100, floor=65, ceiling=1100):
    """audio：单声道；shift_cents：frame_rate 网格上的移调量。返回同长波形。"""
    changed = np.abs(shift_cents) > 1e-3
    if not changed.any():
        return np.asarray(audio, dtype=np.float32).copy()
    shift_cents, blend = _controls(shift_cents, sr, len(audio), frame_rate)
    if not np.any(np.abs(shift_cents) > 1e-3):
        return np.asarray(audio, dtype=np.float32).copy()
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", 0.01, floor, ceiling)
    source = call(manipulation, "Extract pitch tier")
    target = call("Create PitchTier", "target", sound.xmin, sound.xmax)
    # 补尾部零点，使最后一帧也连续回到原音高。
    grid = np.arange(len(shift_cents)+1) / frame_rate
    shift_cents = np.r_[shift_cents, 0.]
    for i in range(1, call(source, "Get number of points") + 1):
        t = call(source, "Get time from index", i)
        hz = call(source, "Get value at index", i)
        cents = float(np.interp(t, grid, shift_cents, left=0., right=0.))
        call(target, "Add point", t, hz * 2 ** (cents / 1200))
    call([target, manipulation], "Replace pitch tier")
    out = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    out = np.pad(out, (0, max(0, len(audio) - len(out))))[:len(audio)]
    # 逐采样平滑混合，避免每 10ms 跳档；零移调区仍精确旁路。
    out = audio + (out - audio) * blend
    return out.astype(np.float32)
