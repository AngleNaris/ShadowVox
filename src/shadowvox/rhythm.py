"""节奏对齐：以**原唱入点**为基准，把干声的时间轴轻轻拉回来。

设计依据（2026-10-23，用户明确）：
  - 目标是**跟原唱对齐**，不是跟伴奏对齐。原唱是唯一基准。
  - 优先级：**入点 > 中间 > 出点**。听感上最刺耳的就是「晚半个字进来」。
  - 底线：宁可节奏不准也要自然 → 只做**平滑、限速**的时间伸缩，不做逐字硬贴。

漂移怎么测（这个方法被严格标定过）：
  把两轨的浊音起点做成高斯脉冲序列，滑窗互相关。
  用「原唱整体平移 d」标定：7 组真值误差全部 0.0ms，各窗 std 0.0。
  比之前所有方案都干净：
    ✗ onset 能量包络互相关（真值 0ms 测出 -260ms）
    ✗ 10fps / 100fps DTW（路径贴死带宽，结果是常数）
    ✗ 最近邻起点配对（154 个点只配上 88 个，配对错位产生 ±1 音节假抖动）

实测（青花瓷）：局部漂移 P50=44ms / P90=166ms / 34% 的窗口超 80ms。

怎么改：
  漂移曲线 → 平滑 + 限速 → WSOLA 变时（保音高）→ 交给后面的音高流程。
  曲线本身由 6s 窗口估出，天然平滑，伸缩率长期贴近 1.0，不会出现逐字掰的机械感。
"""
import numpy as np

from .align import voiced_onsets

FPS = 100.0


def onset_train(f0, blur=3):
    """浊音起点 → 单位面积高斯脉冲序列（100fps 网格）。"""
    n = len(f0)
    t = np.zeros(n, dtype=np.float32)
    on = voiced_onsets(f0).astype(int)
    ok = (on >= 0) & (on < n)
    t[on[ok]] = 1.0
    k = np.exp(-0.5 * (np.arange(-blur, blur + 1) / blur) ** 2)
    return np.convolve(t, k / k.sum(), mode="same")


def onset_offset(dry_f0, ref_f0, max_lag=1.5):
    """起音点脉冲序列互相关，求全局偏移（正 = 干声比原唱晚）。

    标定：原唱整体平移 ±200ms 共 7 组，误差全部 0.0ms。
    比「音高误差扫描」更贴合入点优先级 —— 音高扫描对齐的是音符中段。
    """
    a, b = onset_train(dry_f0), onset_train(ref_f0)
    if not (a.any() and b.any()):
        return 0.
    ml = int(max_lag * FPS)
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.arange(-ml, ml + 1)
    vals = cc[np.mod(lags, size)]
    k = int(np.argmax(vals))
    frac = 0.0
    if 0 < k < len(vals) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return -(lags[k] + frac) / FPS


def _key_invariant_pitch_error(dry_f0, ref_f0, shift_frames, min_frames=120):
    """移调无关的音高一致性评估：将 dry[j] 与 ref[j + shift_frames] 对比。

    自动搜索最优半音移调 k ∈ [-5, +6]，返回 (中位音分误差, 120音分内占比, 最优半音, 共同有声帧数)。
    既兼容同调演唱，也能在 KTV 升降调 / 男女异调翻唱时准确辨别真实对齐位置。
    """
    from .f0 import cent

    nd, nr = len(dry_f0), len(ref_f0)
    idx = np.arange(nd, dtype=np.int64) + int(shift_frames)
    ok = (idx >= 0) & (idx < nr) & (dry_f0 > 0)
    if int(ok.sum()) < min_frames:
        return np.inf, 0.0, 0, int(ok.sum())
    rv = ref_f0[np.clip(idx, 0, nr - 1)]
    m = ok & (rv > 0)
    cnt = int(m.sum())
    if cnt < min_frames:
        return np.inf, 0.0, 0, cnt
    diff = (cent(dry_f0[m]) - cent(rv[m]) + 600.0) % 1200.0 - 600.0
    semis = np.arange(-5, 7, dtype=np.float64)[:, None] * 100.0
    folded = np.abs(((diff[None, :] - semis + 600.0) % 1200.0) - 600.0)
    meds = np.median(folded, axis=1)
    best_i = int(np.argmin(meds))
    return float(meds[best_i]), float(np.mean(folded[best_i] <= 120.0)), int(best_i - 5), cnt


def _top_xcorr_peaks(a, b, max_lag_frames, k=6, min_sep_frames=50):
    """求 corr[lag] = Σ_t b[t + lag] * a[t] 的前 k 个间距峰，lag 语义为 ref - dry（帧）。"""
    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    ml = min(int(max_lag_frames), size // 2 - 1)
    if ml <= 0:
        return []
    lags = np.arange(-ml, ml + 1)
    vals = cc[np.mod(lags, size)]
    order = np.argsort(vals)[::-1]
    peaks = []
    for idx in order:
        lag = int(lags[idx])
        if all(abs(lag - p[0]) >= min_sep_frames for p in peaks):
            frac = 0.0
            if 0 < idx < len(vals) - 1:
                y0, y1, y2 = vals[idx - 1], vals[idx], vals[idx + 1]
                den = y0 - 2.0 * y1 + y2
                if abs(den) > 1e-9:
                    frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            peaks.append((lag + frac, float(vals[idx])))
            if len(peaks) >= k:
                break
    return peaks


def wide_onset_offset(dry_f0, ref_f0, max_lag=12.0):
    """宽范围全局偏移估计（正 = 干声比原唱晚）：移调无关，兼容 ±12s 前奏/录音延迟。"""
    from .align import COARSE_FPS, _decimate, _estimate_offset

    near = onset_offset(dry_f0, ref_f0, max_lag=1.5)
    if len(dry_f0) < 200 or len(ref_f0) < 200:
        return near
    a, b = onset_train(dry_f0), onset_train(ref_f0)
    if not (a.any() and b.any()):
        return near
    # 直接在 ±max_lag 全范围搜索移调无关的起音脉冲互相关峰值，并补充色度粗搜候选。
    candidates = [-near]  # 候选均按 ref - dry（秒）表示
    for lag_f, _ in _top_xcorr_peaks(a, b, int(round(max_lag * FPS)), k=6, min_sep_frames=50):
        c = lag_f / FPS
        if all(abs(c - prev) > 0.15 for prev in candidates):
            candidates.append(c)
    d10 = _decimate(dry_f0, FPS, COARSE_FPS)
    r10 = _decimate(ref_f0, FPS, COARSE_FPS)
    coarse_ref_minus_dry = -float(np.clip(_estimate_offset(d10, r10, COARSE_FPS), -max_lag, max_lag))
    if abs(coarse_ref_minus_dry) > 1.2 and all(abs(coarse_ref_minus_dry - prev) > 0.4 for prev in candidates):
        size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
        cc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
        center = int(round(coarse_ref_minus_dry * FPS))
        rad = int(round(1.0 * FPS))
        lags = np.arange(center - rad, center + rad + 1)
        vals = cc[np.mod(lags, size)]
        k = int(np.argmax(vals))
        frac = 0.0
        if 0 < k < len(vals) - 1:
            y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-9:
                frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        candidates.append((lags[k] + frac) / FPS)

    near_err, near_in, _, near_n = _key_invariant_pitch_error(
        dry_f0, ref_f0, int(round(-near * FPS)), min_frames=200)
    best_ref_minus_dry = -near
    best_err = near_err
    for cand in candidates[1:]:
        err, in_key, _, cnt = _key_invariant_pitch_error(
            dry_f0, ref_f0, int(round(cand * FPS)), min_frames=200)
        if not np.isfinite(err) or cnt < 200:
            continue
        # 仅当远处候选明显优于 ±1.5s 近处峰（至少低 25 音分且旋律一致）时采纳。
        if (not np.isfinite(best_err) or err + 25.0 < best_err) and (err <= 130.0 or in_key >= 0.55):
            best_ref_minus_dry = cand
            best_err = err
    return -float(best_ref_minus_dry)


def _track_offset_windows(dry_f0, ref_f0, pred_fn, radius_s=2.5, win_s=20.0, step_s=10.0):
    nd, nr = len(dry_f0), len(ref_f0)
    win_f = int(round(win_s * FPS))
    step_f = int(round(step_s * FPS))
    rad_f = int(round(radius_s * FPS))
    pts = []
    for i0 in range(0, max(1, nd - win_f // 2), step_f):
        i1 = min(nd, i0 + win_f)
        if i1 - i0 < int(round(10.0 * FPS)):
            continue
        d_seg = dry_f0[i0:i1]
        if int((d_seg > 0).sum()) < 150:
            continue
        t_mid = 0.5 * (i0 + i1) / FPS
        exp_lag = float(pred_fn(t_mid))
        j0 = max(0, i0 + int(round((exp_lag - radius_s) * FPS)))
        j1 = min(nr, i1 + int(round((exp_lag + radius_s) * FPS)))
        if j1 - j0 < int(round(8.0 * FPS)):
            continue
        r_seg = ref_f0[j0:j1]
        if int((r_seg > 0).sum()) < 150:
            continue
        a, b = onset_train(d_seg), onset_train(r_seg)
        if not (a.any() and b.any()):
            continue
        exp_loc = i0 + exp_lag * FPS - j0
        max_loc = int(abs(exp_loc) + rad_f + 10)
        peaks = _top_xcorr_peaks(a, b, max_loc, k=4, min_sep_frames=20)
        best = None
        for loc_lag, val in peaks:
            if abs(loc_lag - exp_loc) > rad_f + 5:
                continue
            rmd_f = (j0 - i0) + loc_lag
            mad, in_key, key_k, _ = _key_invariant_pitch_error(
                d_seg, ref_f0, int(round(rmd_f + i0)), min_frames=100)
            if not np.isfinite(mad):
                continue
            if mad <= 115.0 and in_key >= 0.55:
                cand = (mad - 25.0 * in_key, rmd_f / FPS, key_k, mad, in_key, val)
                if best is None or cand[0] < best[0]:
                    best = cand
        if best is not None:
            pts.append((t_mid, best[1], best[2], best[3], best[4], best[5]))
    return pts


def estimate_affine_alignment(dry_f0, ref_f0, max_lag=12.0, max_scale_dev=0.04):
    """移调无关的全局仿射对齐估计：t_ref ≈ scale * t_dry + offset。

    通过滑窗起音互相关 + 移调无关旋律校验提取全曲多窗时差，自动区分：
      1) 恒定录音偏移（scale = 1.0，offset = 恒定时差）；
      2) 不同版本伴奏/母带导致的微小线性变速漂移（如 0.4% 速度差导致首尾差 600ms）。
    只有全曲 ≥4 窗、跨度 ≥40s 且线性残差远低于恒定残差时才启用非 1.0 速度比。
    """
    empty = dict(
        scale=1.0, offset=0.0, reliable=False, scale_applied=False,
        inliers=0, total_windows=0, span_seconds=0.0,
        const_residual_ms=0.0, linear_residual_ms=0.0, drift_ms=0.0, key_semitones=0,
    )
    if dry_f0 is None or ref_f0 is None or len(dry_f0) < 200 or len(ref_f0) < 200:
        return empty
    a, b = onset_train(dry_f0), onset_train(ref_f0)
    if not (a.any() and b.any()):
        return empty
    peaks = _top_xcorr_peaks(a, b, int(round(max_lag * FPS)), k=5, min_sep_frames=60)
    centers = [p[0] / FPS for p in peaks]
    wo = -wide_onset_offset(dry_f0, ref_f0, max_lag=max_lag)
    if all(abs(wo - c) > 0.35 for c in centers):
        centers.insert(0, wo)
    if all(abs(c) > 0.5 for c in centers):
        centers.append(0.0)

    best_pts = []
    for c in centers[:4]:
        pts = _track_offset_windows(dry_f0, ref_f0, lambda _t, _c=c: _c, radius_s=2.5)
        if len(pts) > len(best_pts):
            best_pts = pts
    if len(best_pts) < 3:
        return empty

    keys = np.array([p[2] for p in best_pts])
    best_key = int(max(
        range(-5, 7),
        key=lambda k: (
            int(np.sum(((keys - k + 6) % 12) - 6 == 0)),
            int(np.sum(np.abs(((keys - k + 6) % 12) - 6) <= 1)),
            -abs(k),
        ),
    ))
    pts = [p for p in best_pts if abs(((p[2] - best_key + 6) % 12) - 6) <= 1]
    if len(pts) < 3:
        return empty

    ts = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    slopes = [
        (ys[j] - ys[i]) / (ts[j] - ts[i])
        for i in range(len(ts)) for j in range(i + 1, len(ts))
        if ts[j] - ts[i] >= 15.0 and abs((ys[j] - ys[i]) / (ts[j] - ts[i])) <= max_scale_dev
    ]
    if slopes:
        s0 = float(np.median(slopes))
        b0 = float(np.median(ys - s0 * ts))
        if abs(s0) >= 0.0005:
            pts2 = _track_offset_windows(dry_f0, ref_f0, lambda t, _s=s0, _b=b0: _s * t + _b, radius_s=1.2)
            pts2 = [p for p in pts2 if abs(((p[2] - best_key + 6) % 12) - 6) <= 1]
            if len(pts2) >= len(pts):
                pts = pts2
                ts = np.array([p[0] for p in pts], dtype=np.float64)
                ys = np.array([p[1] for p in pts], dtype=np.float64)

    c_med = float(np.median(ys))
    c_err = np.abs(ys - c_med)
    c_res = float(np.median(c_err))
    c_inliers = c_err <= 0.12
    c_count = int(c_inliers.sum())
    c_span = float(np.ptp(ts[c_inliers])) if c_count >= 2 else 0.0

    slopes = [
        (ys[j] - ys[i]) / (ts[j] - ts[i])
        for i in range(len(ts)) for j in range(i + 1, len(ts))
        if ts[j] - ts[i] >= 15.0 and abs((ys[j] - ys[i]) / (ts[j] - ts[i])) <= max_scale_dev
    ]
    scale_applied = False
    slope, intercept, lin_res, lin_span, lin_n = 0.0, c_med, c_res, c_span, c_count
    if slopes:
        s0 = float(np.median(slopes))
        b0 = float(np.median(ys - s0 * ts))
        inliers = np.abs(ys - (s0 * ts + b0)) <= 0.06
        if int(inliers.sum()) >= 4:
            slope, intercept = [float(v) for v in np.polyfit(ts[inliers], ys[inliers], 1)]
            inliers = np.abs(ys - (slope * ts + intercept)) <= 0.06
            if int(inliers.sum()) >= 4:
                slope, intercept = [float(v) for v in np.polyfit(ts[inliers], ys[inliers], 1)]
                lin_n = int(inliers.sum())
                lin_span = float(np.ptp(ts[inliers]))
                lin_res = float(np.median(np.abs(ys[inliers] - (slope * ts[inliers] + intercept))))
                drift = abs(slope) * lin_span
                if (
                    lin_n >= 4
                    and lin_n / len(ts) >= 0.70
                    and lin_span >= 40.0
                    and 0.0008 <= abs(slope) <= max_scale_dev
                    and drift >= 0.12
                    and lin_res <= 0.030
                    and lin_res <= 0.45 * max(c_res, 1e-6)
                ):
                    scale_applied = True

    if scale_applied:
        return dict(
            scale=1.0 + slope, offset=intercept, reliable=True, scale_applied=True,
            inliers=lin_n, total_windows=len(ts), span_seconds=round(lin_span, 2),
            const_residual_ms=round(c_res * 1000.0, 2),
            linear_residual_ms=round(lin_res * 1000.0, 2),
            drift_ms=round(abs(slope) * lin_span * 1000.0, 2),
            key_semitones=best_key,
        )
    reliable = bool(c_count >= 3 and c_count / len(ts) >= 0.70 and c_span >= 20.0 and c_res <= 0.06)
    off = float(np.median(ys[c_inliers])) if reliable else 0.0
    return dict(
        scale=1.0, offset=off, reliable=reliable, scale_applied=False,
        inliers=c_count, total_windows=len(ts), span_seconds=round(c_span, 2),
        const_residual_ms=round(c_res * 1000.0, 2),
        linear_residual_ms=round(lin_res * 1000.0, 2),
        drift_ms=round(abs(slope) * lin_span * 1000.0, 2),
        key_semitones=best_key,
    )


def onset_alignment(dry_f0, ref_f0, global_offset, skip=0.08, scale=1.0):
    """把两轨的浊音起点做**单调对齐**（允许跳过任一侧），返回 (干声秒, 偏差秒)。

    偏差 = (原唱位置 + global_offset) - scale * 干声位置（正 = 干声进早了）。
    单调约束保证不会出现「第 5 个字对到第 4 个字」这种交叉错配。
    之前用最近邻配对：154 个点只配上 88 个，错配还会伪造出 ±1 个音节的假抖动
    （误以为相邻偏差差 110ms，实际只有 60ms）。

    代价用 1D 时序 DP：匹配代价 = |Δt|，跳过代价 = skip（秒）。
    """
    td = voiced_onsets(dry_f0) / FPS
    tr = voiced_onsets(ref_f0) / FPS
    n, m = len(td), len(tr)
    if n == 0 or m == 0:
        return np.zeros(0), np.zeros(0)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    D[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, c = INF, 0
            if i > 0 and j > 0:
                v = D[i - 1, j - 1] + abs(scale * td[i - 1] - (tr[j - 1] + global_offset))
                if v < best:
                    best, c = v, 0
            if i > 0 and D[i - 1, j] + skip < best:
                best, c = D[i - 1, j] + skip, 1
            if j > 0 and D[i, j - 1] + skip < best:
                best, c = D[i, j - 1] + skip, 2
            D[i, j], P[i, j] = best, c
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        c = P[i, j]
        if c == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif c == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    if not pairs:
        return np.zeros(0), np.zeros(0)
    ta = np.array([td[a] for a, _ in pairs])
    dv = np.array([(tr[b] + global_offset) - scale * td[a] for a, b in pairs])
    return ta, dv


def correction_curve(ta, dv, max_rate=0.10, smooth_win=0):
    """由入点偏差生成整条时间轴的修正曲线 c(t)：内容需要**提前**多少秒。

    c = -dv（干声进早 dv>0 → 要往后挪，即提前量为负）。
    曲线在入点处精确穿过，入点之间线性；再做限速，
    保证 |dc/dt| <= max_rate，即局部伸缩率落在 [1-max_rate, 1+max_rate]。

    这样「入点」被钉住（优先），音符中间由线性过渡决定，出点自然不被硬掰 ——
    正好对应优先级：入点 > 中间 > 出点。

    ⚠️ **不要开平滑**。实测 smooth_win=1 就把曲线打成半折：
    平均残差 10ms→43ms，达成率 0.87→0.52。原因是相邻入点的偏差正负交替，
    平滑会把它们互相抵消掉。
    """
    if len(ta) == 0:
        return lambda t: 0.0
    order = np.argsort(ta)
    ta, dv = ta[order], dv[order]
    # 合并过近的锚点，避免线性段过短导致伸缩率爆炸
    keep = [0]
    for k in range(1, len(ta)):
        if ta[k] - ta[keep[-1]] >= 0.20:
            keep.append(k)
    ta, dv = ta[keep], dv[keep]
    c = -dv.astype(np.float64)

    if smooth_win >= 1 and len(c) >= 3:
        ker = np.ones(smooth_win * 2 + 1) / (smooth_win * 2 + 1)
        c = np.convolve(np.pad(c, smooth_win, mode="edge"), ker, mode="same")[smooth_win:-smooth_win]

    for i in range(1, len(c)):
        step = max_rate * max(1e-6, ta[i] - ta[i - 1])
        c[i] = float(np.clip(c[i], c[i - 1] - step, c[i - 1] + step))
    for i in range(len(c) - 2, -1, -1):
        step = max_rate * max(1e-6, ta[i + 1] - ta[i])
        c[i] = float(np.clip(c[i], c[i + 1] - step, c[i + 1] + step))

    def curve(t):
        return float(np.interp(t, ta, c, left=c[0], right=c[-1]))

    return curve


def warp_wsola(x, sr, lead, n=None, frame=1024, hop=256, tol_s=0.008):
    """按「提前量」曲线做保音高变时（WSOLA）。

    lead(t_out)：该输出时刻应取自源音频的提前量（秒，正 = 用后面的内容）。
    输出与输入等长，采样级对齐，不做整体移位。

    tol（±8ms）内用归一化互相关找相位最连贯的接点，避免拼接产生周期性碰撞音。
    """
    x = np.asarray(x, dtype=np.float32)
    n = n or len(x)
    out = np.zeros(n + frame + 2 * hop, dtype=np.float32)
    wt = np.zeros_like(out)
    w = np.hanning(frame).astype(np.float32)
    tol = max(1, int(tol_s * sr))
    half = frame // 2
    prev = None

    for t_out in range(0, n, hop):
        p = t_out + lead(t_out / sr) * sr
        pos = int(round(p))
        if prev is None:
            src = pos
        else:
            lo = max(0, pos - tol)
            hi = min(len(x) - half - 1, pos + tol)
            if hi <= lo:
                src = pos
            else:
                seg = x[lo:hi + half].astype(np.float64)
                seg = seg - seg.mean()
                # np.correlate(a, v, 'valid') 的第 k 项 = Σ a[k+i]·v[i]
                cc = np.correlate(seg, prev, mode="valid")
                src = lo + int(np.argmax(cc))
        src = max(0, min(src, max(0, len(x) - frame)))
        seg = x[src:src + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        out[t_out:t_out + frame] += seg * w
        wt[t_out:t_out + frame] += w
        ov = out[t_out + hop:t_out + hop + half]
        if len(ov) >= half:
            t = ov[:half].astype(np.float64)
            t = t - t.mean()
            nn = np.sqrt((t * t).sum())
            prev = t / nn if nn > 1e-9 else t
    # 窗和归一（hop=frame/4 时窗和约为 2，必须除掉，否则电平翻倍）
    nz = wt > 1e-6
    out[nz] /= wt[nz]
    return out[:n]


def align_legacy(dry, dry_f0, ref_f0, sr, max_rate=0.10, log=print):
    """一站式：求入点最优全局偏移 → 单调配对 → 建曲线 → WSOLA 变时。

    返回 (新音频, 统计)。变时后**偏移会变**，调用方应在新波形上重测 offset。
    """
    g = onset_offset(dry_f0, ref_f0)
    ta, dv = onset_alignment(dry_f0, ref_f0, g)
    if len(ta) < 3:
        return dry, {"onset_pairs": int(len(ta)), "applied": False}
    # 再按「偏差中位数为 0」微调一次全局偏移（中位 = 系统性偏置）
    g = g + float(np.median(dv))
    ta, dv = onset_alignment(dry_f0, ref_f0, g)
    if len(ta) < 3:
        return dry, {"onset_pairs": int(len(ta)), "applied": False}

    curve = correction_curve(ta, dv, max_rate=max_rate)
    y = warp_wsola(dry, sr, curve, n=len(dry))
    a = np.abs(dv)
    stat = {
        "onset_pairs": int(len(ta)),
        "onset_offset_before_seconds": round(float(g), 4),
        "onset_dev_p50_ms": round(float(np.percentile(a, 50)) * 1000, 1),
        "onset_dev_p90_ms": round(float(np.percentile(a, 90)) * 1000, 1),
        "onset_dev_max_ms": round(float(a.max()) * 1000, 1),
        "max_rate": max_rate,
        "applied": True,
    }
    log(f"[节奏] 入点 {len(ta)} 对，偏差 P50={stat['onset_dev_p50_ms']:.0f}ms "
        f"P90={stat['onset_dev_p90_ms']:.0f}ms → 限速 {max_rate:.0%} 已对齐")
    return y, stat


def align_to_reference(dry, dry_f0, ref_f0, sr, max_rate=0.10, log=print):
    """默认有限修正；旧版本仅保留用于回归对照。"""
    from .guarded_rhythm import align
    return align(dry, dry_f0, ref_f0, sr, max_rate=max_rate, log=log)
