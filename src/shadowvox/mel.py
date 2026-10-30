"""Slaney 梅尔滤波器组与对数幅度谱，按 OpenTune 的 MelSpectrogram.cpp 规则复刻。"""
import numpy as np

_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOG_STEP = np.log(6.4) / 27.0


def hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(hz < _MIN_LOG_HZ, hz / _F_SP,
                    _MIN_LOG_MEL + np.log(np.maximum(hz, 1e-10) / _MIN_LOG_HZ) / _LOG_STEP)


def mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel < _MIN_LOG_MEL, _F_SP * mel,
                    _MIN_LOG_HZ * np.exp(_LOG_STEP * (mel - _MIN_LOG_MEL)))


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """三角形滤波器，Slaney 归一化，在精确 FFT bin 频率上采样（不做整数吸附）。"""
    n_bins = n_fft // 2 + 1
    eff_fmax = min(float(fmax), 0.5 * float(sr))
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(eff_fmax)
    idx = np.arange(1, n_mels + 1, dtype=np.float64)
    centers = mel_to_hz(mel_min + (mel_max - mel_min) * idx / (n_mels + 1))
    knots = np.concatenate(([float(fmin)], centers, [eff_fmax]))

    freqs = np.arange(n_bins, dtype=np.float64) * (float(sr) / float(n_fft))
    w = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = knots[m], knots[m + 1], knots[m + 2]
        left_span, right_span = center - left, right - center
        if left_span <= 0 or right_span <= 0:
            continue
        enorm = 2.0 / (right - left)
        lower = (freqs - left) / left_span
        upper = (right - freqs) / right_span
        tri = np.minimum(lower, upper)
        tri[tri <= 0] = 0.0
        w[m] = (tri * enorm).astype(np.float32)
    return w


def _frames(x, win, hop, n_frames):
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def log_mel(x, sr, n_fft, win, hop, n_mels, fmin, fmax, log_eps=1e-5, n_frames=None):
    """返回 [T, n_mels] 的自然对数幅度梅尔谱。中心对称反射填充，周期性 Hann 窗。"""
    x = np.asarray(x, dtype=np.float32)
    pad_left = (win - hop) // 2
    pad_right = (win - hop + 1) // 2
    npad = pad_left + pad_right
    if len(x) > npad:
        padded = np.pad(x, (pad_left, pad_right), mode="reflect")
    else:
        padded = np.pad(x, (pad_left, pad_right), mode="edge")

    if n_frames is None:
        n_frames = 1 + (len(padded) - win) // hop
    n_frames = max(1, min(n_frames, 1 + (len(padded) - win) // hop))

    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win) / win))
    frames = _frames(padded, win, hop, n_frames) * hann[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)

    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = mag @ fb.T
    return np.log(np.maximum(mel, log_eps)).astype(np.float32)
