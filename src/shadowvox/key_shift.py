"""自动异调检测与高质量伴奏/参考移调：适配 KTV 升降调与男女换调翻唱。"""
import numpy as np

from .config import F0_FRAME_RATE
from .f0 import cent


def _folded_abs(diff_cents, shift_cents):
    wrapped = ((diff_cents - shift_cents + 600.0) % 1200.0) - 600.0
    return np.abs(wrapped)


def detect_key_shift(dry_f0, ref_f0, *, reference_times=None, reference_valid=None,
                     offset_seconds=0.0, fps=F0_FRAME_RATE):
    """检测干声相对原唱是否存在稳定的整半音调性偏移（[-5, +6] 半音）。

    严格区分「异调翻唱」与「同调跑调」：要求移调后全曲误差大幅收敛、大部分乐句一致，
    且对单半音偏移启用更严苛门限，避免误把略偏高/偏低的同调演唱判成异调。
    """
    dry_f0 = np.asarray(dry_f0, dtype=np.float64)
    ref_f0 = np.asarray(ref_f0, dtype=np.float64)
    nd, nr = len(dry_f0), len(ref_f0)
    empty = {
        "applied": False,
        "semitones": 0,
        "before_error_cents": None,
        "after_error_cents": None,
        "before_in_key_ratio": None,
        "after_in_key_ratio": None,
        "consistency": 0.0,
        "matched_frames": 0,
    }
    if nd < 50 or nr < 50:
        return empty

    if reference_times is not None and reference_valid is not None:
        ref_map = np.asarray(reference_times, dtype=np.float64) * fps
        valid = (np.asarray(reference_valid, dtype=bool) & np.isfinite(ref_map)
                 & (ref_map >= 0.0) & (ref_map <= nr - 1))
        idx = np.clip(np.rint(np.where(valid, ref_map, 0.0)).astype(np.int64), 0, nr - 1)
    else:
        lag = int(round(float(offset_seconds) * fps))
        raw_idx = np.arange(nd, dtype=np.int64) - lag
        valid = (raw_idx >= 0) & (raw_idx < nr)
        idx = np.clip(raw_idx, 0, nr - 1)

    ok = valid & (dry_f0 > 0.0) & (ref_f0[idx] > 0.0)
    matched = int(ok.sum())
    if matched < 150:
        empty["matched_frames"] = matched
        return empty

    diff = cent(dry_f0[ok]) - cent(ref_f0[idx[ok]])

    def evaluate(semi):
        ad = _folded_abs(diff, semi * 100.0)
        return float(np.median(ad)), float(np.mean(ad <= 100.0))

    err0, in0 = evaluate(0)
    best_k, best_err, best_in = 0, err0, in0
    for k in range(-5, 7):
        if k == 0:
            continue
        ek, ik = evaluate(k)
        if ek < best_err - 1e-6 or (abs(ek - best_err) <= 1e-6 and abs(k) < abs(best_k)):
            best_k, best_err, best_in = k, ek, ik

    # 按时间顺序切分 8 段校验全曲一致性，防止单句跑偏劫持整曲
    chunks = [ch for ch in np.array_split(diff, 8) if len(ch) >= 15]
    agree = sum(1 for ch in chunks if float(np.median(_folded_abs(ch, best_k * 100.0))) <= 120.0)
    consistency = agree / max(1, len(chunks))

    triggered = (
        best_k != 0
        and best_err <= 95.0
        and (err0 - best_err) >= 90.0
        and best_err <= 0.55 * err0
        and best_in >= 0.60
        and (best_in - in0) >= 0.25
        and consistency >= 0.65
    )
    if triggered and abs(best_k) == 1:
        triggered = (
            err0 >= 85.0
            and best_err <= 48.0
            and best_in >= 0.75
            and (best_in - in0) >= 0.40
            and consistency >= 0.75
        )

    return {
        "applied": bool(triggered),
        "semitones": int(best_k) if triggered else 0,
        "before_error_cents": round(err0, 1),
        "after_error_cents": round(best_err if triggered else err0, 1),
        "before_in_key_ratio": round(in0, 3),
        "after_in_key_ratio": round(best_in if triggered else in0, 3),
        "consistency": round(float(consistency), 3),
        "matched_frames": matched,
    }


def transpose_f0(f0, semitones):
    """按整半音平移 F0 曲线，保持清音帧为 0。"""
    arr = np.asarray(f0, dtype=np.float32).copy()
    if not semitones:
        return arr
    ratio = float(2.0 ** (semitones / 12.0))
    mask = arr > 0.0
    arr[mask] *= ratio
    return arr


def transpose_audio(samples, sr, semitones):
    """使用 pedalboard 高质量变调移调伴奏，严格保持原时长与通道数。"""
    arr = np.asarray(samples, dtype=np.float32)
    if not semitones or arr.size == 0:
        return arr.copy()
    from pedalboard import PitchShift

    mono = arr.ndim == 1
    buf = arr[None, :] if mono else arr
    shifted = PitchShift(semitones=float(semitones))(buf.astype(np.float32), sr)
    shifted = np.asarray(shifted, dtype=np.float32)
    n = buf.shape[1]
    if shifted.shape[1] < n:
        shifted = np.pad(shifted, ((0, 0), (0, n - shifted.shape[1])))
    elif shifted.shape[1] > n:
        shifted = shifted[:, :n]
    shifted = np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
    return shifted[0] if mono else shifted
