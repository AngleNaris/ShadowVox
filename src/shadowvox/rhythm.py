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


def onset_alignment(dry_f0, ref_f0, global_offset, skip=0.08):
    """把两轨的浊音起点做**单调对齐**（允许跳过任一侧），返回 (干声秒, 偏差秒)。

    偏差 = 原唱位置 - 干声位置（正 = 干声进早了）。
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
                v = D[i - 1, j - 1] + abs(td[i - 1] - (tr[j - 1] + global_offset))
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
    dv = np.array([(tr[b] + global_offset) - td[a] for a, b in pairs])
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
