"""PC-NSF-HiFiGAN 声码器：mel + f0 → 波形。分块合成 + 交叉淡化拼接。"""
import numpy as np
import onnxruntime as ort

from . import config as C
from .mel import log_mel

CHUNK_FRAMES = 1024      # 11.9s
OVERLAP_FRAMES = 32      # 0.37s


def _default_providers():
    avail = ort.get_available_providers()
    for p in ("DmlExecutionProvider", "CUDAExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def vocoder_mel(x, n_frames=None):
    """x: 44.1k 单声道音频 → [T,128] 对数梅尔（幅度谱）。"""
    if n_frames is None:
        n_frames = max(1, len(x) // C.VOC_HOP)
    return log_mel(x, C.VOC_SR, C.VOC_FFT, C.VOC_WIN, C.VOC_HOP,
                   C.VOC_MELS, C.VOC_FMIN, C.VOC_FMAX, C.MEL_LOG_EPS, n_frames)


def map_f0_to_frames(f0, n_frames, f0_frame_rate=C.F0_FRAME_RATE):
    """把 100fps 的 f0 曲线映射到声码器帧网格（对 logarithm 域插值）。"""
    pos = np.arange(n_frames, dtype=np.float64) * C.VOC_HOP / C.VOC_SR * f0_frame_rate
    pos = np.clip(pos, 0, len(f0) - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(f0) - 1)
    frac = pos - i0
    a, b = f0[i0], f0[i1]
    valid = (a > 0) & (b > 0)
    out = np.where(valid, np.exp(np.log(np.maximum(a, 1e-6)) * (1 - frac)
                                 + np.log(np.maximum(b, 1e-6)) * frac), 0.0)
    out = np.where(valid, out, np.where(a > 0, a, b))
    nearest = np.rint(pos).astype(np.int64)
    voiced = f0[nearest] > 0
    out = np.where(voiced, out, 0.0)
    return out.astype(np.float32)


class Vocoder:
    def __init__(self, onnx_path=C.VOCODER_ONNX, providers=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=providers or _default_providers())
        names = [i.name for i in self.sess.get_inputs()]
        self.mel_name = next(n for n in names if "mel" in n.lower())
        self.f0_name = next(n for n in names if "f0" in n.lower() or "pitch" in n.lower())

    def _run(self, mel, f0):
        y = self.sess.run(None, {self.mel_name: mel[None], self.f0_name: f0[None]})[0]
        return y.reshape(-1)

    def synth(self, audio_44k, f0, progress=None):
        """audio_44k: 44.1k 单声道原始人声；f0: 100fps 目标音高。返回同长波形。"""
        audio_44k = np.asarray(audio_44k, dtype=np.float32)
        n_samples = len(audio_44k)
        n_frames = max(1, n_samples // C.VOC_HOP)
        mel = vocoder_mel(audio_44k, n_frames)
        f0f = map_f0_to_frames(f0, n_frames)

        out = np.zeros((n_frames + OVERLAP_FRAMES) * C.VOC_HOP, dtype=np.float32)
        weight = np.zeros_like(out)

        step = CHUNK_FRAMES - OVERLAP_FRAMES * 2
        total = max(1, (n_frames + step - 1) // step)
        for ci, start in enumerate(range(0, n_frames, step)):
            lo = max(0, start - OVERLAP_FRAMES)
            hi = min(n_frames, start + CHUNK_FRAMES - OVERLAP_FRAMES)
            if hi - lo < 8:
                break
            y = self._run(mel[lo:hi], f0f[lo:hi])
            s0 = lo * C.VOC_HOP
            s1 = min(len(out), s0 + len(y))
            y = y[: s1 - s0]

            w = np.ones(len(y), dtype=np.float32)
            fade = min(OVERLAP_FRAMES * C.VOC_HOP, len(y) // 2)
            if fade > 0:
                if lo > 0:
                    w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
                if hi < n_frames:
                    w[-fade:] = np.minimum(w[-fade:], np.linspace(1, 0, fade, dtype=np.float32))
            out[s0:s1] += y * w
            weight[s0:s1] += w
            if progress:
                progress((ci + 1) / total)

        nz = weight > 1e-6
        out[nz] /= weight[nz]
        out = out[:n_samples]
        if len(out) < n_samples:
            out = np.pad(out, (0, n_samples - len(out)))
        return out
