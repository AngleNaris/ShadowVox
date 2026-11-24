"""半曲/局部演唱自动末尾缓冲与整曲平滑淡出。"""
import numpy as np


def _last_active_seconds(signal, sr, floor_rel_db=-36.0):
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.sqrt(np.mean(np.square(arr), axis=0))
    if not arr.size or sr <= 0:
        return 0.0
    hop = max(1, int(round(0.05 * sr)))
    count = len(arr) // hop
    if count < 2:
        return len(arr) / sr if float(np.max(np.abs(arr))) > 1e-4 else 0.0
    frames = arr[:count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    peak = float(np.percentile(rms, 95))
    thresh = max(1e-4, peak * (10.0 ** (floor_rel_db / 20.0)))
    idx = np.flatnonzero(rms >= thresh)
    return float((idx[-1] + 1) * hop / sr) if idx.size else 0.0


def plan_partial_fadeout(placed_vocal, ref_vocal, sr, *,
                         buffer_seconds=3.5, fade_seconds=4.5,
                         min_ref_remaining_seconds=12.0, min_song_remaining_seconds=15.0):
    """检测干声是否只唱了前半段；若是，则在干声末尾预留缓冲并规划平滑淡出终点。

    同时要求原唱仍有 >=12s 未唱完、且伴奏剩余 >=15s，避免误裁全曲的自然纯乐器尾奏。
    """
    song_samples = len(placed_vocal)
    song_seconds = song_samples / float(sr) if sr > 0 else 0.0
    last_vocal_s = _last_active_seconds(placed_vocal, sr)
    ref_last_s = _last_active_seconds(ref_vocal, sr)
    ref_remaining = ref_last_s - last_vocal_s
    song_remaining = song_seconds - last_vocal_s

    target_end_s = min(song_seconds, last_vocal_s + float(buffer_seconds) + float(fade_seconds))
    applied = (
        last_vocal_s > 1.0
        and ref_remaining >= float(min_ref_remaining_seconds)
        and song_remaining >= float(min_song_remaining_seconds)
        and target_end_s < song_seconds - 2.0
    )
    end_samples = int(round(target_end_s * sr)) if applied else song_samples
    end_samples = max(1, min(song_samples, end_samples))
    return {
        "applied": bool(applied),
        "last_vocal_seconds": round(last_vocal_s, 2),
        "ref_last_vocal_seconds": round(ref_last_s, 2),
        "buffer_seconds": float(buffer_seconds) if applied else 0.0,
        "fade_seconds": float(fade_seconds) if applied else 0.0,
        "fade_start_seconds": round(last_vocal_s + float(buffer_seconds), 2) if applied else None,
        "end_seconds": round(end_samples / float(sr), 2) if sr > 0 else 0.0,
        "end_samples": int(end_samples),
        "original_song_seconds": round(song_seconds, 2),
    }


def apply_fadeout(samples, sr, fade_seconds):
    """对音频末尾施加半余弦平滑淡出，无咔哒声。"""
    arr = np.asarray(samples, dtype=np.float32).copy()
    fade_n = int(round(float(fade_seconds) * sr))
    length = arr.shape[-1]
    if fade_n <= 1 or length <= 1:
        return arr
    fade_n = min(fade_n, length)
    t = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    curve = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    if arr.ndim == 1:
        arr[-fade_n:] *= curve
    else:
        arr[..., -fade_n:] *= curve
    return arr
