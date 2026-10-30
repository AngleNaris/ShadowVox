"""TD-PSOLA 渲染（Praat / parselmouth）：直接在原始波形上按音分曲线移调。

不重新生成人声，气声、辅音和共振峰保留原样；只按 Praat 自己分析的音高点做相对移调，
不需要修的位置音高不变。质量依赖 Praat 的音高周期标注，严重误检处可能出现瑕疵。
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.ndimage import distance_transform_edt


def render(audio, sr, shift_cents, frame_rate=100, floor=65, ceiling=1100):
    """audio：单声道；shift_cents：frame_rate 网格上的移调量。返回同长波形。"""
    changed = np.abs(shift_cents) > 1e-3
    if not changed.any():
        return np.asarray(audio, dtype=np.float32).copy()
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", 0.01, floor, ceiling)
    source = call(manipulation, "Extract pitch tier")
    target = call("Create PitchTier", "target", sound.xmin, sound.xmax)
    grid = np.arange(len(shift_cents)) / frame_rate
    for i in range(1, call(source, "Get number of points") + 1):
        t = call(source, "Get time from index", i)
        hz = call(source, "Get value at index", i)
        cents = float(np.interp(t, grid, shift_cents, left=0., right=0.))
        call(target, "Add point", t, hz * 2 ** (cents / 1200))
    call([target, manipulation], "Replace pitch tier")
    out = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    out = np.pad(out, (0, max(0, len(audio) - len(out))))[:len(audio)]
    # 零移调、清音与对齐保护区逐采样旁路；只在实际修音区域内交叉淡化。
    blend = np.clip(distance_transform_edt(changed) / 3., 0., 1.)
    frame = np.minimum((np.arange(len(audio)) * frame_rate / sr).astype(int), len(blend) - 1)
    out = audio + (out - audio) * blend[frame]
    return out.astype(np.float32)
