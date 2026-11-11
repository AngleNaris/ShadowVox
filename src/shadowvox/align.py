"""干声 ↔ 原唱的单调对齐与音高映射。

思路：不做音符级硬贴，而是
  1) 半音级色度(chroma)互相关估整体时移（解决录音延迟/去头）；
  2) 在时移后的重叠区做窄带 DTW（10fps）修残余漂移，得到单调逐帧映射；
  3) 原唱音高按八度折叠到干声音域；
  4) shift = 目标 - 干声，再对 shift 曲线做中值+高斯平滑（保留干声自身颤音）；
  5) 置信度衰减 + 最大移调钳位。

整体时移怎么定（2026-10-23 推翻旧做法）：
色度互相关在 10fps 网格上**分不清 0 和 ±100ms** ——《青花瓷》上 lag=+1 与 lag=0
的归一化分只差 11%（1.0000 vs 0.8910），抛物线插值也补不回来。实测它给出的
-0.086s 与真值 +0.008s 差 94ms，而且《过火》上连**方向都反了**（+0.299 vs -0.030）。

所以现在：色度只在 ±12s 内做**粗搜**（分辨整段错位/去头），
再在粗搜结果附近 ±0.6s 用「音高误差中位」做**精扫**（2ms 步长，逐帧比 15000 帧，
能分辨到 2ms 级）。音高扫描是唯一与听感一致的判据：
青花瓷 +8ms → 47 音分，-86ms → 108 音分；过火 -30ms → 64 音分，+299ms → 190 音分。
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from .f0 import cent, from_cent

COARSE_FPS = 10.0
F0_FPS = 100.0
MAX_OFFSET_SECONDS = 12.0


@dataclass
class ShiftResult:
    shift_cents: np.ndarray          # 100fps，与干声等长
    confidence: np.ndarray           # 100fps
    offset_seconds: float            # 干声比原唱晚多少（正=晚）。pipeline 取负号后交给 _place
    stats: dict = field(default_factory=dict)


def _decimate(f0, src_fps, dst_fps):
    step = int(round(src_fps / dst_fps))
    if step <= 1:
        return f0.astype(np.float64).copy()
    n = len(f0) // step
    if n == 0:
        return np.zeros(0)
    seg = f0[: n * step].reshape(n, step).astype(np.float64)
    voiced = seg > 0
    out = np.zeros(n)
    ok = voiced.any(axis=1)
    if ok.any():
        masked = np.where(voiced[ok], seg[ok], np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanmedian(masked, axis=1)
        out[ok] = np.nan_to_num(vals, nan=0.0)
    return out


def _chroma(cents, voiced, n_bins=12):
    """软色度：按 10 音分分辨率撒到最近的半音，清音帧全零。"""
    t = np.zeros((len(cents), n_bins), dtype=np.float32)
    if len(cents) == 0:
        return t
    semi = cents / 100.0
    idx = np.mod(np.rint(semi).astype(np.int64), n_bins)
    frac = semi - np.rint(semi)
    w = np.exp(-0.5 * (frac / 0.35) ** 2).astype(np.float32)
    rows = np.nonzero(voiced)[0]
    t[rows, idx[rows]] = 1.0
    prev = np.mod(idx - 1, n_bins)
    nxt = np.mod(idx + 1, n_bins)
    t[rows, prev[rows]] += 0.35 * w[rows]
    t[rows, nxt[rows]] += 0.35 * w[rows]
    return t


def _pitch_error_at(dry_f0, ref_f0, shift_frames):
    """干声整体平移 shift_frames（100fps 帧）后，与 ref 的逐帧音分误差中位。

    shift_frames 语义 = **干声需要移动的量**（负 = 提前，正 = 延后）。
    验证：「干声内容延后 150ms」在 shift=-15 帧处误差 0.00 音分，±10 帧升到 35 音分。
    """
    nd = len(dry_f0)
    idx = np.arange(nd, dtype=np.int64) + int(shift_frames)
    ok = (idx >= 0) & (idx < len(ref_f0)) & (dry_f0 > 0)
    if ok.sum() < 300:
        return np.inf, 0
    rv = ref_f0[np.clip(idx, 0, len(ref_f0) - 1)]
    m = ok & (rv > 0)
    if m.sum() < 300:
        return np.inf, int(m.sum())
    d = np.abs(cent(dry_f0[m]) - cent(rv[m]))
    d = np.minimum(np.mod(d, 1200.0), 1200.0 - np.mod(d, 1200.0))
    return float(np.median(d)), int(m.sum())


def _refine_by_pitch(dry_f0, ref_f0, center_s=None, radius_s=0.60, step_frames=1):
    """用音高误差做全局精扫，返回 (干声比原唱晚多少秒, 音分误差)。

    center_s 只用来定搜索中心；给了则搜 ±radius_s，不给则搜 ±MAX_REFINE_SECONDS。
    粗扫用 10 帧步长（峰宽只有约 20 帧，20 帧步长会跨过峰顶 —— 踩过），
    再在峰附近 1 帧步长细化，最后抛物线取亚帧。

    比色度可靠得多：色度在 10fps 网格上分不清 0 与 ±100ms，甚至判错方向
    （《青花瓷》-86ms vs 真值 +80ms，《过火》+299ms vs 真值 -30ms）。

    符号：_pitch_error_at 的 shift 是「干声需移动量」，本函数返回「干声晚多少」，
    两者互为相反数，所以最后取负。（已用「延后 150ms → 返回 +0.150」标定。）
    """
    c = int(round((center_s or 0.0) * F0_FPS))
    rad = int(radius_s * F0_FPS)
    coarse = [(sh, _pitch_error_at(dry_f0, ref_f0, sh)[0])
              for sh in range(c - rad, c + rad + 1, 10)]
    coarse = [(sh, e) for sh, e in coarse if np.isfinite(e)]
    if not coarse:
        return 0.0, np.inf
    best_sh = min(coarse, key=lambda x: x[1])[0]
    fine = [(sh, _pitch_error_at(dry_f0, ref_f0, sh)[0])
            for sh in range(best_sh - 12, best_sh + 13, step_frames)]
    fine = [(sh, e) for sh, e in fine if np.isfinite(e)]
    if not fine:
        return 0.0, np.inf
    k = int(np.argmin([e for _, e in fine]))
    best_sh, best_e = fine[k]
    if 0 < k < len(fine) - 1:                # 抛物线亚帧
        y0, y1, y2 = fine[k - 1][1], fine[k][1], fine[k + 1][1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            best_sh = best_sh + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * step_frames
    return -best_sh / F0_FPS, best_e


def _estimate_offset(dry_f0, ref_f0, fps):
    """粗估整体时移（色度互相关），返回秒。**只用于给精扫一个起点。**

    语义：**正 = 干声比原唱晚**。色度在 10fps 网格上精度只有 ±100ms，
    甚至可能连方向都判错，所以必须再走 _refine_by_pitch。
    """
    dc, rc = cent(dry_f0), cent(ref_f0)
    d_ch = _chroma(dc, dry_f0 > 0)
    r_ch = _chroma(rc, ref_f0 > 0)
    if d_ch.sum() < 1 or r_ch.sum() < 1:
        return 0.0

    # 两轨必须补到同一长度 n，否则零填充位置错开会让 lag 偏移 len(r)-len(d)
    n = max(len(d_ch), len(r_ch))
    max_lag = int(MAX_OFFSET_SECONDS * fps)
    size = 1 << int(np.ceil(np.log2(2 * n + 1)))
    score = np.zeros(2 * max_lag + 1)
    for b in range(12):
        a = np.zeros(n, dtype=np.float32)   # 参考轨
        c = np.zeros(n, dtype=np.float32)   # 干声轨
        a[: len(r_ch)] = r_ch[:, b]
        c[: len(d_ch)] = d_ch[:, b]
        # corr[s] = Σ_t r[t] * d[t-s]
        fa = np.fft.rfft(a, n=size)
        fc = np.fft.rfft(c, n=size)
        corr = np.fft.irfft(fa * np.conj(fc), n=size)
        score += corr[np.mod(np.arange(-max_lag, max_lag + 1), size)]
    lags = np.arange(-max_lag, max_lag + 1)
    # 轻微偏好小时移，避免对到远处重复段
    score = score / (1.0 + 0.02 * np.abs(lags))
    k = int(np.argmax(score))
    frac = 0.0
    if 0 < k < len(score) - 1:          # 抛物线插值取亚帧精度
        y0, y1, y2 = score[k - 1], score[k], score[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return -(float(lags[k]) + frac) / fps


def _banded_dtw(c_dry, c_ref, v_dry, v_ref, band_frames, diag_prior=0.03):
    """窄带 DTW（两段等长），返回 ref 索引映射（与 dry 等长）。"""
    n = min(len(c_dry), len(c_ref))
    if n < 4:
        return np.arange(n, dtype=np.float64)
    pc_d, pc_r = np.mod(c_dry[:n], 1200.0), np.mod(c_ref[:n], 1200.0)
    d = np.abs(pc_r[:, None] - pc_d[None, :])
    dist = np.clip(np.minimum(d, 1200.0 - d) / 300.0, 0.0, 1.0)
    cost = dist + 0.5 * np.abs(v_ref[:n, None].astype(np.float32)
                              - v_dry[None, :n].astype(np.float32))

    band = max(4, int(band_frames))
    ii = np.arange(n)[:, None]
    jj = np.arange(n)[None, :]
    gap = np.abs(ii - jj)
    cost = cost + diag_prior * gap          # 对角先验：优先不弯曲
    cost = np.where(gap > band, 1e6, cost).astype(np.float32)

    # 三步 DTW；只访问带内格点，避免桌面包为这一操作携带 librosa/Numba JIT。
    cumulative = np.full((n+1,n+1),np.inf)
    cumulative[0,0] = 0
    parent = np.zeros((n,n),dtype=np.uint8)
    for i in range(n):
        for j in range(max(0,i-band),min(n,i+band+1)):
            options = (cumulative[i,j],cumulative[i+1,j],cumulative[i,j+1])
            step = int(np.argmin(options))
            cumulative[i+1,j+1] = options[step]+float(cost[i,j])
            parent[i,j] = step
    i,j=n-1,n-1
    wp=[]
    while True:
        wp.append((i,j))
        if i==0 and j==0:
            break
        step=parent[i,j]
        if step in (0,2):i-=1
        if step in (0,1):j-=1
    path = np.asarray(wp)
    if path.ndim != 2:
        raise RuntimeError("DTW 路径格式异常")
    # librosa<0.10 返回 (2,k)，>=0.10 返回 (k,2)
    if path.shape[0] == 2 and path.shape[1] != 2:
        path_r, path_d = path[0], path[1]
    else:
        path_r, path_d = path[:, 0], path[:, 1]
    path_r = path_r[::-1].astype(np.float64)
    path_d = path_d[::-1].astype(np.float64)
    _, uniq = np.unique(path_d, return_index=True)
    uniq = np.sort(uniq)
    return np.interp(np.arange(n, dtype=np.float64), path_d[uniq], path_r[uniq])


def estimate_shift(dry_f0, ref_f0, strength=1.0, max_shift_cents=700.0,
                   smooth_ms=0.0, band_seconds=0.30, offset_override=None,
                   pitch_mode="frame", reference_times=None, reference_valid=None, register_detail=0.):
    nd = len(dry_f0)
    if nd == 0 or len(ref_f0) == 0:
        raise ValueError("音高曲线为空")
    fps = COARSE_FPS
    d10 = _decimate(dry_f0, F0_FPS, fps)
    r10 = _decimate(ref_f0, F0_FPS, fps)

    # offset = 干声比原唱晚多少（正=晚）。干声晚 lag 秒 ⇒ dry[j] 对应 ref[j - lag]，
    # 即 ref 侧下标为 j - lag_frames，重叠区也据此裁剪。
    offset_coarse = _estimate_offset(d10, r10, fps)
    if offset_override is not None:
        # 外部已定好偏移（例如节奏对齐给出的入点偏移）时，必须用它，
        # 否则音高映射与最终摆放错开，音准会明显变差（实测 24→40 音分）。
        offset = float(offset_override)
        pitch_err = _pitch_error_at(dry_f0, ref_f0, -int(round(offset * F0_FPS)))[0]
    else:
        # 色度精度只有 ±100ms（分不清 0 与 ±1 格），且可能判错方向 ——
        # 直接在 ±1.5s 内用音高误差全局精扫，不再受色度起点约束。
        offset, pitch_err = _refine_by_pitch(dry_f0, ref_f0, None, radius_s=1.5)
    if reference_times is None:
        ref_map = _pitch_reference_map(d10, r10, nd, offset, band_seconds)
        trusted = None
    else:
        ref_map = np.asarray(reference_times, dtype=np.float64) * F0_FPS
        if ref_map.shape != (nd,) or reference_valid is None or np.shape(reference_valid) != (nd,):
            raise ValueError("外部音高对应需要等长的参考时标与可信掩码")
        trusted = (np.asarray(reference_valid, dtype=bool) & np.isfinite(ref_map) &
                   (ref_map >= 0) & (ref_map <= len(ref_f0)-1))
        furthest = np.maximum.accumulate(np.where(trusted, ref_map, -np.inf))
        trusted &= ref_map >= furthest - 1e-6

    ref_c = cent(ref_f0)
    safe_map = np.clip(np.where(np.isfinite(ref_map), ref_map, 0.), 0, len(ref_f0)-1)
    idx = np.rint(safe_map).astype(np.int64)
    target = ref_c[idx]
    dry_voiced = dry_f0 > 0
    reference_voiced = ref_f0[idx] > 0
    voiced = dry_voiced & reference_voiced
    if trusted is not None:
        voiced &= trusted

    dry_c = cent(dry_f0)
    octave = np.rint((dry_c - target) / 1200.0)
    target_folded = target + 1200.0 * octave
    raw = np.where(voiced, target_folded - dry_c, 0.0)

    pc_gap = np.mod(np.abs(np.mod(target, 1200.0) - np.mod(dry_c, 1200.0)), 1200.0)
    pc_gap = np.minimum(pc_gap, 1200.0 - pc_gap)
    confidence = np.clip(1.0 - (pc_gap - 80.0) / 220.0, 0.0, 1.0)

    register_stats = {}
    if pitch_mode == "register":
        if trusted is None:
            from .register import register_shift
            shift, reliable, octave = register_shift(dry_c, target, voiced, strength,
                                                      min(max_shift_cents, 450), register_detail)
        else:
            from .register import correspondence_shift
            shift, reliable, octave, coverage = correspondence_shift(
                dry_c, target, voiced, strength, min(max_shift_cents, 450), register_detail)
            coverage.update(voiced_input_frames=int(dry_voiced.sum()),
                            unmatched_reference_frames=int((dry_voiced & ~trusted).sum()),
                            reference_unvoiced_frames=int((dry_voiced & trusted & ~reference_voiced).sum()),
                            matched_voiced_frames=int(voiced.sum()))
            register_stats['register_coverage'] = coverage
        confidence = reliable.astype(float)
        register_stats.update(register_skipped_frames=int((voiced & ~reliable).sum()),
                              register_octave_counts={str(int(k)): int(np.sum(octave[voiced] == k))
                                                      for k in np.unique(octave[voiced])})
    elif pitch_mode == "natural":
        from .natural import center_shift
        shift = center_shift(dry_c, target_folded, voiced, strength, max_shift_cents)
    elif pitch_mode == "frame":
        shift = _smooth(raw, int(smooth_ms / 1000.0 * F0_FPS))
        shift = np.clip(shift, -max_shift_cents, max_shift_cents) * confidence * float(strength)
    else:
        raise ValueError(f"未知音高模式: {pitch_mode}")
    shift[~voiced] = 0.0
    confidence[~voiced] = 0.0

    stats = {
        "offset_seconds": offset,
        "offset_coarse_seconds": offset_coarse,
        "pitch_error_cents": round(pitch_err, 1) if np.isfinite(pitch_err) else None,
        "aligned_frames": int(voiced.sum()),
        "voiced_ratio": float(voiced.mean()) if nd else 0.0,
        "mean_abs_shift_cents": float(np.abs(shift[voiced]).mean()) if voiced.any() else 0.0,
        "max_abs_shift_cents": float(np.abs(shift).max()) if nd else 0.0,
        "mean_confidence": float(confidence[voiced].mean()) if voiced.any() else 0.0,
        "octave_folded_frames": int(np.count_nonzero(octave[voiced])),
        **register_stats,
    }
    return ShiftResult(shift_cents=shift, confidence=confidence,
                       offset_seconds=offset, stats=stats)


def _pitch_reference_map(d10, r10, nd, offset, band_seconds):
    """没有独立对应路径时，沿用原有 F0 匹配。"""
    fps = COARSE_FPS
    lag_frames = int(round(offset * fps))
    # DTW 只能在整数帧上裁窗口，但**映射时必须用小数偏移**：
    # 否则 <100ms 的偏移会被量化掉一半（实测残差 50ms，音准 24→40 音分）。
    lag_exact = offset * fps

    # 重叠区（干声下标 j）：ref 侧下标为 j - lag_frames
    j0 = max(0, lag_frames)
    j1 = min(len(d10), len(r10) + lag_frames)
    if j1 - j0 < 4:
        raise ValueError("干声与原曲没有可对齐的重叠区")

    c_d, c_r = cent(d10[j0:j1]), cent(r10[j0 - lag_frames: j1 - lag_frames])
    v_d, v_r = d10[j0:j1] > 0, r10[j0 - lag_frames: j1 - lag_frames] > 0
    local_map = _banded_dtw(c_d, c_r, v_d, v_r, band_frames=band_seconds * fps)

    # 拼成干声域（10fps）的全局映射：dry_idx -> ref_idx
    # 段内下标 a = j - j0；DTW 返回的 local_map[a] 已是段内 ref 下标，
    # 换算到全局：ref10 = j0 - lag_exact + local_map[a]
    grid = np.arange(len(d10), dtype=np.float64)
    ref_map10 = np.full(len(d10), np.nan)
    ref_map10[j0:j1] = j0 - lag_exact + local_map
    valid = ~np.isnan(ref_map10)
    if valid.sum() < 4:
        raise ValueError("对齐失败，可用帧不足")
    # 端点外推用最近有效值，避免 np.interp 默认线性外推把索引带飞
    ref_map10 = np.interp(grid, grid[valid], ref_map10[valid],
                          left=ref_map10[valid][0], right=ref_map10[valid][-1])
    ref_map10 = np.clip(ref_map10, 0, len(r10) - 1)

    # 10fps 帧号 → 100fps 帧号：只缩放、不再叠加时间原点
    t10 = np.arange(nd, dtype=np.float64) / F0_FPS          # 秒
    ref_map = np.interp(t10, grid / fps, ref_map10) * (F0_FPS / fps)

    return ref_map


def voiced_onsets(f0, min_dur=0.06):
    """浊音段起点（100fps 帧号）。这些点直接对应「字头开始振动」，是节奏对齐的锚点。

    已用「把原唱平移 10ms」标定：onset 位置精确移动 1 帧，噪声底为 0 帧。
    """
    v = np.asarray(f0) > 0
    d = np.diff(v.astype(np.int8))
    st = np.nonzero(d == 1)[0] + 1
    en = np.nonzero(d == -1)[0] + 1
    if v[0]:
        st = np.r_[0, st]
    if v[-1]:
        en = np.r_[en, len(v)]
    pairs = [(s, e) for s, e in zip(st, en) if (e - s) >= min_dur * F0_FPS]
    return np.array([s for s, _ in pairs], dtype=np.float64)


def _smooth(x, win):
    if win < 3 or len(x) < win:
        return x
    from scipy.ndimage import gaussian_filter1d, median_filter
    y = median_filter(x, size=max(3, win // 2), mode="nearest")
    return gaussian_filter1d(y, sigma=max(1.0, win / 4.0), mode="nearest")


def apply_shift(dry_f0, shift_cents):
    out = np.zeros_like(dry_f0, dtype=np.float32)
    v = dry_f0 > 0
    out[v] = from_cent(cent(dry_f0[v]) + np.asarray(shift_cents)[v])
    return out
