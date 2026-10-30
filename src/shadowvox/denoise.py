"""在修音前对干声做温和的 DeepFilterNet3 降噪。"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import audio, config as C


def process(samples, sr):
    """返回与输入长度一致、时间位置不变的单声道人声。"""
    samples = np.asarray(samples, dtype=np.float32)
    if not samples.size:
        return samples.copy()
    helper = (C.ROOT / "bin" / "shadowvox-denoise.exe" if getattr(sys, "frozen", False)
              else C.ROOT / "tools" / "vendor" / "deepfilternet" / "shadowvox-denoise.exe")
    if not helper.is_file():
        raise RuntimeError("未找到 AI 降噪组件，请重新安装完整桌面程序。")

    # scipy 的多相重采样补偿滤波延迟；往返后裁回原始采样数。
    from math import gcd
    divisor = gcd(int(sr), 48000)
    up, down = 48000 // divisor, int(sr) // divisor
    at_48k = resample_poly(samples, up, down).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="shadowvox-denoise-") as folder:
        source, target = Path(folder) / "input.npy", Path(folder) / "output.npy"
        np.save(source, at_48k)
        env = os.environ.copy()
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        result = subprocess.run([str(helper), str(source), str(target)], capture_output=True,
                                env=env, **audio.subprocess_options())
        if result.returncode:
            detail = result.stderr.decode(errors="replace").strip()[-800:]
            raise RuntimeError(f"AI 降噪失败：{detail or '组件异常退出'}")
        cleaned = np.load(target, allow_pickle=False)
    restored = resample_poly(cleaned, down, up).astype(np.float32)
    if len(restored) < len(samples):
        restored = np.pad(restored, (0, len(samples) - len(restored)))
    return restored[:len(samples)]
