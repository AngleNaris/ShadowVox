"""Rubber Band R3：官方命令行工具（GPL-2.0-or-later）。桌面版内置于 bin/，源码环境需另行下载。"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import subprocess_options
from .config import F0_FRAME_RATE

ROOT = Path(__file__).resolve().parents[2]
# Rubber Band 4.0 的 CLI 在 44.1 kHz 实测变调提前约 80 ms。
_PITCH_MAP_DELAY_FRAMES = round(.080 * F0_FRAME_RATE)
_PADDING_FRAMES = round(.200 * F0_FRAME_RATE)


def executable():
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin" / "rubberband.exe"
    if getattr(sys, "frozen", False) and bundled.exists():
        return str(bundled)
    local = ROOT / "tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows/rubberband.exe"
    found = str(local) if local.exists() else shutil.which("rubberband")
    if not found:
        raise RuntimeError("未找到 Rubber Band 4.0 命令行工具，请安装并加入 PATH")
    return found


def render(x, shift, sr=44100, *, checkpoint=None):
    """连续渲染并保留共振峰，首尾补零让短音的移调指令完整执行。"""
    shift = np.round(shift, 2)
    if not np.any(shift):
        return x.copy(), 0
    exe = executable()
    padding = round(sr * _PADDING_FRAMES / F0_FRAME_RATE)
    shift = np.pad(shift, (_PADDING_FRAMES + _PITCH_MAP_DELAY_FRAMES,
                           _PADDING_FRAMES - _PITCH_MAP_DELAY_FRAMES))
    with tempfile.TemporaryDirectory(prefix="shadowvox-rb-") as tmp:
        tmp = Path(tmp)
        sf.write(tmp / "input.wav", np.pad(x, (padding, padding)), sr, subtype="FLOAT")
        commands = []
        previous = None
        for i, cents in enumerate(shift):
            if cents != previous:
                commands.append(f"{round(i * sr / F0_FRAME_RATE)} {cents / 100:.6f}")
                previous = cents
        (tmp / "pitchmap.txt").write_text("\n".join(commands), encoding="ascii")
        if checkpoint is not None:
            checkpoint()
        with subprocess.Popen([exe, "-3", "-F", "-t", "1", "--pitchmap", "pitchmap.txt",
                               "-q", "input.wav", "output.wav"], cwd=tmp, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, **subprocess_options()) as process:
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
                raise RuntimeError(stderr.decode(errors="replace"))
        y, _ = sf.read(tmp / "output.wav", dtype="float32")
        length_delta = len(y) - len(x) - 2 * padding
        if abs(length_delta) > sr // 10:
            raise RuntimeError(f"Rubber Band 长度偏差超过 100ms: {length_delta}")
        return y[padding:padding + len(x)].copy(), length_delta
