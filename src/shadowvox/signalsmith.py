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


def render_pitch(x, shift, sr, *, f0=None, checkpoint=None):
    if not np.any(np.round(shift, 2)):
        return x.copy()
    if len(shift) == 1:
        shift = np.repeat(shift, 2)
    points = np.c_[np.arange(len(shift)) * sr / F0_FRAME_RATE, shift]
    voiced = f0[np.isfinite(f0) & (f0 > 0)] if f0 is not None else ()
    fundamental = float(np.median(voiced)) if len(voiced) else 0.
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
