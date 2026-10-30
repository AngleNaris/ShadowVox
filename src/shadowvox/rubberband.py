"""Rubber Band R3：官方命令行工具（GPL-2.0-or-later）。桌面版内置于 bin/，源码环境需另行下载。"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import subprocess_options

ROOT = Path(__file__).resolve().parents[2]


def executable():
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin" / "rubberband.exe"
    if getattr(sys, "frozen", False) and bundled.exists():
        return str(bundled)
    local = ROOT / "tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows/rubberband.exe"
    found = str(local) if local.exists() else shutil.which("rubberband")
    if not found:
        raise RuntimeError("需要 Rubber Band 4.0 命令行工具，安装路径见 README 自然度验证")
    return found


def render(x, shift, sr=44100):
    """官方 R3 工具的逐采样位置 pitchmap，保共振峰，不分段切音频。"""
    exe = executable()
    with tempfile.TemporaryDirectory(prefix="shadowvox-rb-") as tmp:
        tmp = Path(tmp)
        sf.write(tmp / "input.wav", x, sr, subtype="FLOAT")
        commands = []
        previous = None
        for i, cents in enumerate(np.round(shift, 2)):
            if cents != previous:
                commands.append(f"{round(i * sr / 100)} {cents / 100:.6f}")
                previous = cents
        (tmp / "pitchmap.txt").write_text("\n".join(commands), encoding="ascii")
        p = subprocess.run([exe, "-3", "-F", "-t", "1", "--pitchmap", "pitchmap.txt",
                            "-q", "input.wav", "output.wav"],
                           cwd=tmp, capture_output=True, **subprocess_options())
        if p.returncode:
            raise RuntimeError(p.stderr.decode(errors="replace"))
        y, _ = sf.read(tmp / "output.wav", dtype="float32")
        length_delta = len(y) - len(x)
        if abs(length_delta) > sr // 10:
            raise RuntimeError(f"Rubber Band 长度偏差超过 100ms: {length_delta}")
        return np.pad(y, (0, max(0, len(x)-len(y))))[:len(x)], length_delta
