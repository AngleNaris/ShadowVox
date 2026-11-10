"""Fixed C+ finalizer. Section shaping and limiter drive follow ShadowBuster."""
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import sosfilt

from . import audio, config as C
from .density import densify
from .vst_worker import render_isolated

TARGET_LUFS = -8.0
TRUE_PEAK_CEILING = -0.5
MAX_MEDIAN_REDUCTION_DB = 3.2
LOUDNESS_TOLERANCE = .15
LIMITER = C.ROOT/'plugins/ZL Limiter.vst3/Contents/x86_64-win/ZL Limiter.vst3'


def _block_loudness(mixed, sr, seconds):
    # Digital K-weighting as used by ShadowBuster's BS.1770 fallback.
    a = 10 ** (4.0 / 40)
    w = 2 * np.pi * 1500 / sr
    c, beta = np.cos(w), np.sqrt(2 * a) * np.sin(w)
    b = np.array([a*(a+1+(a-1)*c+beta), -2*a*(a-1+(a+1)*c), a*(a+1+(a-1)*c-beta)])
    d = np.array([a+1-(a-1)*c+beta, 2*(a-1-(a+1)*c), a+1-(a-1)*c-beta])
    shelf = np.r_[b / d[0], d / d[0]]
    w = 2 * np.pi * 38 / sr
    c, alpha = np.cos(w), np.sin(w)
    b = np.array([(1+c)/2, -(1+c), (1+c)/2])
    d = np.array([1+alpha, -2*c, 1-alpha])
    weighted = sosfilt([shelf, np.r_[b / d[0], d / d[0]]], mixed, axis=-1)
    np.square(weighted, out=weighted)
    block = min(mixed.shape[-1], round(seconds * sr))
    starts = np.arange(0, mixed.shape[-1]-block+1, round(.1*sr))
    power = np.r_[0., np.cumsum(weighted.sum(axis=0))]
    return -.691 + 10*np.log10(np.maximum((power[starts+block]-power[starts])/block, 1e-300))


def _loudness(mixed, sr):
    levels = _block_loudness(mixed, sr, .4)
    active = levels[levels >= -70.]
    if not active.size:
        return -np.inf
    preliminary = 10*np.log10(np.mean(10**(active/10)))
    return float(10*np.log10(np.mean(10**(active[active >= preliminary-10.]/10))))


def _sections(mixed, sr):
    short = median_filter(_block_loudness(mixed, sr, 3.), size=61, mode='nearest')
    active = short[short > -70.]
    if not active.size:
        return np.zeros(mixed.shape[-1], dtype=np.float32)
    anchor = np.percentile(active, 70)
    severity = gaussian_filter1d(np.clip((anchor-3.-short)/6., 0., 1.), sigma=10, mode='nearest')
    centers = min(1.5*sr, mixed.shape[-1]/2) + np.arange(len(severity))*.1*sr
    return np.interp(np.arange(mixed.shape[-1]), centers, severity).astype(np.float32)


def _limit(mixed, sr, gain):
    from pedalboard import load_plugin

    if not LIMITER.is_file():
        raise RuntimeError('缺少 ZL Limiter 0.0.1，请运行 python tools/fetch_zl.py 或使用完整安装包')
    limiter = load_plugin(str(LIMITER))
    limiter.output_ceiling_output_ceiling = TRUE_PEAK_CEILING
    limiter.true_peak_true_peak = True
    limiter.oversampling_oversampling = '4x'
    limiter.channel_delta_channel_delta = 0.0
    frames = mixed.shape[-1]
    padded = np.zeros((2, frames+4096), dtype=np.float32)
    np.multiply(mixed, gain, out=padded[:, :frames])
    result = limiter.process(padded, sr, buffer_size=8192, reset=True)
    if result.shape[0] != 2 or result.shape[1] < frames:
        raise RuntimeError('ZL Limiter 返回的音频不完整')
    return result[:, :frames]


def _window_power(mixed, hop, count):
    windows = mixed[:, :count*hop].reshape(2, count, hop)
    return np.einsum('ijk,ijk->j', windows, windows, dtype=np.float64)/(2*hop)


def finalize(mixed, sr, density=0.):
    """Fit the finished mix to the fixed loudness target, after optional density."""
    if threading.current_thread() is not threading.main_thread():
        return render_isolated('--master-worker', mixed, sr=sr, density=density)
    input_lufs = _loudness(mixed, sr)
    stats = {'engine':'C+', 'target_lufs':TARGET_LUFS, 'true_peak_ceiling_dbtp':TRUE_PEAK_CEILING,
             'input_lufs':input_lufs if np.isfinite(input_lufs) else None, 'master_gain_db':0.,
             'density':{'amount':density,'applied':False}, 'iterations':0, 'stop_reason':'ungated'}
    if not np.isfinite(input_lufs):
        return mixed, stats

    weights = _sections(mixed, sr)
    if density:
        mixed, stats['density'] = densify(mixed, sr, input_lufs, density)
    initial = _loudness(mixed, sr) if stats['density']['applied'] else input_lufs
    if not np.isfinite(initial):
        return mixed, stats
    quiet_trim = 10**(-.5*weights/20.)
    # Measure sustained limiting against the signal actually fed to the limiter.
    hop = min(round(.05*sr), mixed.shape[-1])
    count = mixed.shape[-1]//hop
    power = _window_power(mixed, hop, count)
    active = power > max(1e-12, power.max()*1e-4)
    window_weights = weights[hop//2:count*hop:hop][active]
    input_db = 10*np.log10(power[active])
    drive = float(np.clip(TARGET_LUFS-initial+.75, -24., 24.))
    best, low, high = None, None, None
    constrained = False
    reason = 'iteration_limit'
    for attempt in range(1, 10):
        discount = min(2., .35 * max(drive, 0.))
        candidate = _limit(mixed, sr, 10**((drive-discount*weights)/20.))
        limited_power = _window_power(candidate, hop, count)[active]
        reduction = input_db + drive - discount*window_weights - 10*np.log10(np.maximum(limited_power, 1e-20))
        median_reduction = max(0., float(np.median(reduction))) if reduction.size else 0.
        candidate *= quiet_trim
        actual = _loudness(candidate, sr)
        if not np.isfinite(actual):
            raise RuntimeError('ZL Limiter 返回的响度不可测量')
        error = TARGET_LUFS-actual
        excessive = median_reduction > MAX_MEDIAN_REDUCTION_DB
        if not excessive:
            if best is None or (abs(error), drive) < (best[0], best[2]):
                best = abs(error), candidate, drive, median_reduction
            if abs(error) <= LOUDNESS_TOLERANCE:
                reason = 'fit_converged'
                break
        if excessive or error < 0.:
            high = drive
            constrained = excessive
        else:
            low = drive
        if low is not None and high is not None:
            next_drive = (low+high)/2
        else:
            step = max(.25, min(2.5, abs(error)*2.))
            next_drive = float(np.clip(drive + (step if high is None else -max(step, median_reduction-MAX_MEDIAN_REDUCTION_DB)),
                                       -24., 24.))
        if abs(next_drive-drive) <= .04:
            reason = 'drive_limit'
            break
        drive = next_drive
    if best is None:
        raise RuntimeError('母带在可用增益范围内无法满足持续限幅保护')
    _, result, drive, median_reduction = best
    if constrained and reason != 'fit_converged':
        reason = 'reduction_limit'
    stats.update(master_gain_db=drive, iterations=attempt, stop_reason=reason, limiter_used=True,
                 limiter={'engine':'ZL Limiter 0.0.1','oversampling':4,'true_peak':True,'stereo_link':1.,
                          'median_reduction_db':median_reduction, 'max_median_reduction_db':MAX_MEDIAN_REDUCTION_DB})
    return result, stats


def _written_peak(path):
    process = subprocess.run([audio._ffmpeg_exe(), '-hide_banner', '-nostats', '-i', str(path),
                              '-af', 'aformat=sample_fmts=dbl,aresample=192000,'
                                     'astats=measure_perchannel=none:measure_overall=Peak_level',
                              '-f', 'null', '-'], capture_output=True, check=True, **audio.subprocess_options())
    levels = re.findall(rb'Peak level dB: (-?\d+\.\d+|-inf)', process.stderr)
    if not levels:
        raise RuntimeError('无法回读母带真峰值')
    return float(levels[-1])


def write_output(path, mixed, sr, stats):
    from .mixing import measure_loudness

    path = Path(path)
    correction = 0.
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='master-') as folder:
        pending = Path(folder)/'output.wav'
        audio.save(pending, mixed, sr)
        peak = _written_peak(pending)
        if peak > TRUE_PEAK_CEILING:
            correction = TRUE_PEAK_CEILING-.015-peak
            audio.save(pending, mixed * 10**(correction/20.), sr)
            peak = _written_peak(pending)
        if peak > TRUE_PEAK_CEILING:
            raise RuntimeError('导出的母带超过真峰值上限')
        lufs, _ = measure_loudness(pending)
        pending.replace(path)
    # FFmpeg reports -inf or its measurement floor for ungated output.
    measurable = stats['input_lufs'] is not None and np.isfinite(lufs) and lufs > -70.
    stats.update(estimated_output_lufs=lufs if measurable else None,
                 measured_output_true_peak_dbtp=peak if np.isfinite(peak) else None,
                 true_peak_correction_db=correction, readback_verified=True,
                 target_reached=bool(measurable and abs(lufs-TARGET_LUFS) <= .2))
