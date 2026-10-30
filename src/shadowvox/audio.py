"""音频读写与电平工具：统一用 ffmpeg 解码。"""
import shutil
import sys
import subprocess
import wave
from pathlib import Path

import numpy as np

_FFMPEG = shutil.which("ffmpeg")


def _ffmpeg_exe():
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin" / "ffmpeg.exe"
    if getattr(sys,"frozen",False) and bundled.exists():
        return str(bundled)
    if _FFMPEG:
        return _FFMPEG
    for cand in (r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
        if Path(cand).exists():
            return cand
    raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")


def subprocess_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def load(path, sr=None, channels=None, dtype=np.float32):
    """解码音频。返回 (data, sr)；data 形状 [n] 或 [ch, n]。"""
    path = str(path)
    cmd = [_ffmpeg_exe(), "-v", "error", "-i", path, "-f", "f32le"]
    if sr:
        cmd += ["-ar", str(int(sr))]
    if channels:
        cmd += ["-ac", str(int(channels))]
    cmd += ["-"]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **subprocess_options())
    if process.returncode:
        raise RuntimeError(f"无法读取音频 {path}: {process.stderr.decode(errors='replace')[-600:]}")
    raw = process.stdout
    data = np.frombuffer(raw, dtype="<f4").astype(dtype).copy()
    if data.size == 0:
        raise RuntimeError(f"解码失败或空音频: {path}")

    if channels:
        data = data.reshape(-1, int(channels)).T
    elif sr:
        pass
    else:
        raise ValueError("需要指定 sr 或 channels 以确定声道数")

    if sr is None:
        raise ValueError("必须指定目标采样率")

    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    return np.ascontiguousarray(data), int(sr)


def load_mono(path, sr):
    return load(path, sr=sr, channels=1)[0]


def load_stereo_or_mono(path, sr):
    """尽量保留原始声道数（1 或 2），超过 2 声道下混成立体声。"""
    try:
        data, _ = load(path, sr=sr, channels=2)
    except Exception:
        data, _ = load(path, sr=sr, channels=1)
    return data


def save(path, data, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(arr.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def duration_seconds(path):
    out = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        stdout=subprocess.PIPE, check=True,
    ).stdout.decode().strip()
    return float(out)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def envelope(x, sr, attack=.015, release=.15):
    """10 ms RMS 包络，分别平滑起音和释放。"""
    hop = max(1, round(sr*.01))
    count = (len(x)+hop-1)//hop
    padded = np.pad(x, (0, count*hop-len(x)))
    levels = np.sqrt(np.mean(padded.reshape(-1, hop).astype(np.float64)**2, axis=1)+1e-12)
    smooth = np.zeros_like(levels)
    value = 0.
    for i, level in enumerate(levels):
        coefficient = np.exp(-.01/(attack if level > value else release))
        value = coefficient*value+(1-coefficient)*level
        smooth[i] = value
    return np.interp(np.arange(len(x)), np.arange(count)*hop+hop/2, smooth)


def peak(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def match_rms(x, ref_rms, max_gain_db=18.0):
    cur = rms(x)
    if cur <= 1e-9 or ref_rms <= 1e-9:
        return x
    gain = ref_rms / cur
    lim = 10 ** (max_gain_db / 20.0)
    gain = float(np.clip(gain, 1.0 / lim, lim))
    return x * gain
