"""Signalsmith Stretch 原生助手，共用于音高、变时和实时草稿。"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .audio import subprocess_options
from .config import F0_FRAME_RATE

ENGINE = 'signalsmith'
HELPER_NAME = 'shadowvox-signalsmith.exe'
ROOT = Path(__file__).resolve().parents[2]


def _executable():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS) / 'bin' / HELPER_NAME
    else:
        path = ROOT / 'tools/vendor/signalsmith' / HELPER_NAME
    if not path.is_file():
        raise RuntimeError('缺少 Signalsmith 渲染助手，请运行 python tools/build_signalsmith.py')
    return str(path)


def _render(x, points, mode, sr, length, fundamental=0., checkpoint=None):
    if checkpoint is not None:
        checkpoint()
    command = [_executable(), mode, str(sr), str(fundamental), 'input.f32', 'map.txt', 'output.f32']
    with tempfile.TemporaryDirectory(prefix='shadowvox-signalsmith-') as folder:
        tmp = Path(folder)
        np.asarray(x, dtype='<f4').tofile(tmp/'input.f32')
        np.savetxt(tmp/'map.txt', points, fmt='%.9f')
        with subprocess.Popen(command, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              **subprocess_options()) as process:
            try:
                while True:
                    try:
                        _, stderr = process.communicate(timeout=.1 if checkpoint is not None else None)
                        break
                    except subprocess.TimeoutExpired:
                        checkpoint()
            except InterruptedError:
                process.kill()
                raise
            if process.returncode:
                detail = stderr.decode(errors='replace').strip()
                raise RuntimeError(f'Signalsmith 渲染失败：{detail or process.returncode}')
        result = np.fromfile(tmp/'output.f32', dtype='<f4')
    if len(result) != length or not np.isfinite(result).all():
        raise RuntimeError('Signalsmith 输出长度不符或包含无效采样')
    return result


def _local_f0_track(f0, count, fallback):
    """生成平滑的局部基频轨迹供共振峰保护逐块追踪。"""
    if f0 is None or len(f0) == 0 or fallback <= 0.:
        return np.full(count, fallback, dtype=np.float64)
    arr = np.asarray(f0, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 40.) & (arr < 1600.)
    if not valid.any():
        return np.full(count, fallback, dtype=np.float64)
    idx = np.flatnonzero(valid)
    grid = np.linspace(0., max(0, len(arr) - 1), count) if len(arr) != count else np.arange(count, dtype=np.float64)
    filled = np.interp(grid, idx, arr[idx])
    from scipy.ndimage import median_filter, uniform_filter1d
    smooth = uniform_filter1d(median_filter(filled, size=11, mode='nearest'), size=15, mode='nearest')
    return np.clip(smooth, 65., 1200.)


def smooth_render_shift(shift, f0=None):
    """主流程渲染前平滑：桥接连续有声段内短暂掉零，限制单帧悬崖跳变，消除抖动感。"""
    from scipy.ndimage import gaussian_filter1d

    s = np.asarray(shift, dtype=np.float64).copy()
    n = len(s)
    if n < 3 or not np.any(np.abs(s) > 0.5) or float(np.ptp(s)) < 1e-9:
        return s

    active = np.abs(s) > 0.5
    voiced = (np.isfinite(f0) & (f0 > 0)) if (f0 is not None and len(f0) == n) else np.ones(n, dtype=bool)
    idx = np.flatnonzero(active)
    if len(idx) >= 2:
        gaps = np.flatnonzero(np.diff(idx) > 1)
        for g in gaps:
            a, b = int(idx[g]), int(idx[g + 1])
            if b - a <= 6 and voiced[a:b + 1].all():
                s[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [s[a], s[b]])
                active[a + 1:b] = True

    # 边缘外延 2 帧到清音/静音区，避免 60ms STFT 窗在字尾被陡然拉回 0。
    edges = np.diff(np.r_[0, active.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for st, en in zip(starts, ends):
        for k in (1, 2):
            if st - k >= 0 and not active[st - k]:
                s[st - k] = s[st] * (1.0 - 0.35 * k)
            if en - 1 + k < n and not active[en - 1 + k]:
                s[en - 1 + k] = s[en - 1] * (1.0 - 0.35 * k)

    s = gaussian_filter1d(s, sigma=1.4, mode='nearest')
    max_step = 40.0
    for i in range(1, n):
        s[i] = float(np.clip(s[i], s[i - 1] - max_step, s[i - 1] + max_step))
    for i in range(n - 2, -1, -1):
        s[i] = float(np.clip(s[i], s[i + 1] - max_step, s[i + 1] + max_step))
    s[np.abs(s) < 0.25] = 0.0
    return s


def render_pitch(x, shift, sr, *, f0=None, checkpoint=None):
    if not np.any(np.round(shift, 2)):
        return x.copy()
    if len(shift) == 1:
        shift = np.repeat(shift, 2)
    voiced = f0[np.isfinite(f0) & (f0 > 0)] if f0 is not None else ()
    fundamental = float(np.median(voiced)) if len(voiced) else 0.
    local_f0 = _local_f0_track(f0, len(shift), fundamental)
    points = np.c_[np.arange(len(shift)) * sr / F0_FRAME_RATE, shift, local_f0]
    return _render(x, points, 'pitch', sr, len(x), fundamental, checkpoint)


def stretch(x, source, target, sr, *, checkpoint=None):
    points = np.c_[target, source] * sr
    frames = np.rint(points)
    if (len(frames) < 2 or not np.isfinite(points).all() or np.any(np.diff(frames, axis=0) <= 0)
            or np.any(frames[0] != 0) or frames[-1, 1] > len(x) + 1):
        raise ValueError('变时映射必须从零开始，两侧严格递增，且源时间在音频范围内')
    if np.array_equal(frames[:, 0], frames[:, 1]) and round(points[-1, 0]) == len(x):
        return x.copy()
    return _render(x, points, 'time', sr, round(points[-1, 0]), checkpoint=checkpoint)
