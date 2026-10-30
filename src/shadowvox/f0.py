"""FCPE 音高提取：严格按 OpenTune FCPEExtractor.cpp 复刻。

链路：16k 重采样 → 50Hz 8 阶高通 → -50dBFS 噪声门 → 对数梅尔(1024/160/128)
     → ONNX(latent 360 维) → 置信度阈值 + 质心解码 → f0
"""
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, sosfilt, resample_poly

from . import config as C
from .mel import hz_to_mel, mel_to_hz

_SOS = butter(8, 50.0, btype="highpass", fs=C.F0_SR, output="sos")
_GATE = 0.00316227766
_HANN = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1024) / 1024)).astype(np.float32)
_CENT_TABLE = np.linspace(1200.0 * np.log2(C.F0_MIN / 10.0),
                          1200.0 * np.log2(C.F0_MAX / 10.0), 360).astype(np.float32)


def _fcpe_mel_filterbank():
    """按 FCPE 原实现的整数 bin 吸附 + enorm 归一化构建 [128, 513]。"""
    n_bins = 1024 // 2 + 1
    fsp = 200.0 / 3.0
    min_log_mel = 1000.0 / fsp
    log_step = np.log(6.4) / 27.0

    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(8000.0)
    mel_points = np.linspace(mel_min, mel_max, 130)
    hz_points = mel_to_hz(mel_points)

    freqs = np.arange(n_bins, dtype=np.float64) * C.F0_SR / 1024.0
    w = np.zeros((128, n_bins), dtype=np.float32)
    for m in range(128):
        f_low, f_center, f_high = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        enorm = 2.0 / (f_high - f_low)
        band = (freqs >= f_low) & (freqs <= f_high)
        if not band.any():
            continue
        up = (freqs >= f_low) & (freqs < f_center)
        down = (freqs >= f_center) & (freqs <= f_high)
        w[m, up] = (enorm * (freqs[up] - f_low) / (f_center - f_low)).astype(np.float32)
        w[m, down] = (enorm * (f_high - freqs[down]) / (f_high - f_center)).astype(np.float32)
    return w


_FB = _fcpe_mel_filterbank()


def _preprocess(x):
    y = sosfilt(_SOS, x.astype(np.float64)).astype(np.float32)
    y[np.abs(y) < _GATE] = 0.0
    return y


def _pad(x):
    n = len(x)
    pad_left = (1024 - C.F0_HOP) // 2          # 432
    pad_right = max((1024 - C.F0_HOP + 1) // 2, 1024 - n - pad_left)
    if pad_right < n:
        left = x[1:pad_left + 1][::-1]
        idx = np.maximum(n - 2 - np.arange(pad_right), 0)
        return np.concatenate([left, x, x[idx]])
    return np.concatenate([np.zeros(pad_left, np.float32), x, np.zeros(pad_right, np.float32)])


def _mel(x):
    padded = _pad(x)
    n_frames = max(1, 1 + (len(padded) - 1024) // C.F0_HOP)
    idx = np.arange(1024)[None, :] + C.F0_HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * _HANN[None, :]
    spec = np.fft.rfft(frames, n=1024, axis=1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)
    mel = np.log(np.maximum(mag @ _FB.T, 1e-5))

    target = len(x) // C.F0_HOP + 1
    if mel.shape[0] < target:
        mel = np.vstack([mel, np.repeat(mel[-1:], target - mel.shape[0], axis=0)])
    else:
        mel = mel[:target]
    return mel.astype(np.float32)


def decode_latent(latent, threshold=C.FCPE_CONF_THRESHOLD):
    """[T,360] → f0(Hz)，0 表示清音。argmax + ±4 邻域质心。"""
    peak = latent.max(axis=1)
    idx = latent.argmax(axis=1)
    f0 = np.zeros(latent.shape[0], dtype=np.float32)
    voiced = peak > threshold
    if not voiced.any():
        return f0
    rows = np.nonzero(voiced)[0]
    for t in rows:
        lo, hi = max(0, idx[t] - 4), min(359, idx[t] + 4)
        seg = latent[t, lo:hi + 1]
        ws = float(seg.sum())
        if ws <= 1e-9:
            continue
        cent = float((_CENT_TABLE[lo:hi + 1] * seg).sum() / ws)
        f0[t] = 10.0 * 2.0 ** (cent / 1200.0)
    return f0


class FcpeExtractor:
    def __init__(self, onnx_path=C.FCPE_ONNX, providers=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = providers or _default_providers()
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name

    def extract(self, audio, sr):
        """audio: 任意采样率单声道 → f0 @100fps。"""
        x = np.asarray(audio, dtype=np.float32)
        if sr != C.F0_SR:
            from math import gcd
            g = gcd(int(sr), C.F0_SR)
            x = resample_poly(x, C.F0_SR // g, int(sr) // g).astype(np.float32)
        x = _preprocess(x)
        mel = _mel(x)
        latent = self.sess.run(None, {self.in_name: mel[None]})[0][0]
        f0 = decode_latent(latent)
        f0[(f0 > 0) & (f0 < C.F0_MIN)] = 0.0
        f0[f0 > C.F0_MAX] = C.F0_MAX
        return f0


def _default_providers():
    if C.CPU_ONLY:
        return ['CPUExecutionProvider']
    avail = ort.get_available_providers()
    for p in ("DmlExecutionProvider", "CUDAExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def cent(f0):
    out = np.zeros_like(f0, dtype=np.float64)
    v = f0 > 0
    out[v] = 1200.0 * np.log2(f0[v] / 10.0)
    return out


def from_cent(c):
    return (10.0 * np.power(2.0, np.asarray(c, dtype=np.float64) / 1200.0)).astype(np.float32)
