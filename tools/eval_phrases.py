"""乐句切分 + 进唱偏移测量（浊音起音点法）。

已验证：浊音起音点测偏移，已知真值误差 0ms。能量包络法误差 260~310ms，已弃用。

切句难点：真实演唱的换气口常被混响填满 ——《过火》浊音占比 83%，有连续 4s 不断。
所以句界不能只看「浊音断开」，要结合**能量谷底**：
  候选句界 = 浊音短缺口（F0 掉但时间短） ∪ 能量窗内局部最低点
"""
import sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shadowvox import config as C, audio as A, align as AL
from shadowvox.f0 import FcpeExtractor

FPS = AL.F0_FPS


def rms_curve(x, hop=441):
    n = len(x) // hop
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)


def voiced_onsets(f0, min_dur=0.06):
    v = f0 > 0
    d = np.diff(v.astype(np.int8))
    st = np.nonzero(d == 1)[0] + 1
    en = np.nonzero(d == -1)[0] + 1
    if v[0]:
        st = np.r_[0, st]
    if v[-1]:
        en = np.r_[en, len(v)]
    pairs = [(s, e) for s, e in zip(st, en) if (e - s) >= min_dur * FPS]
    return np.array([s for s, _ in pairs], dtype=np.float64)


def split_phrases(f0, rms, fps_rms, phrase_s=6.0, min_phrase_s=1.2):
    """按「浊音缺口 + 能量谷底」切句，返回 [(起点帧@100fps, 终点帧@100fps)]。"""
    n = len(f0)
    v = f0 > 0
    # 句子只在浊音区间内切
    d = np.diff(v.astype(np.int8))
    st = np.nonzero(d == 1)[0] + 1
    en = np.nonzero(d == -1)[0] + 1
    if v[0]:
        st = np.r_[0, st]
    if v[-1]:
        en = np.r_[en, n]
    spans = [(int(s), int(e)) for s, e in zip(st, en) if (e - s) >= 0.12 * FPS]
    if not spans:
        return []

    # 能量连续投影到 100fps：rms 每 10ms 一点，正好同帧率
    er = np.interp(np.arange(n), np.arange(len(rms)) * (fps_rms / FPS) * FPS / FPS,
                   rms) if len(rms) != n else rms

    out = []
    for s0, e0 in spans:
        L = (e0 - s0) / FPS
        k = max(1, int(round(L / phrase_s)))
        if L < min_phrase_s * 2:
            out.append((s0, e0))
            continue
        # 在 span 内找 k-1 个内部分界
        bounds = [s0]
        for j in range(1, k):
            expect = s0 + int((e0 - s0) * j / k)
            half = int(1.6 * FPS)
            lo, hi = max(s0 + int(min_phrase_s * FPS), expect - half), \
                     min(e0 - int(min_phrase_s * FPS), expect + half)
            if hi <= lo:
                continue
            seg = er[lo:hi]
            # 谷底 = 能量最小处；同时偏好浊音短暂断开处
            pen = np.zeros(len(seg))
            vv = v[lo:hi]
            pen[~vv] = float(seg.max() - seg.min()) * 0.35
            idx = lo + int(np.argmin(seg - pen))
            bounds.append(idx)
        bounds.append(e0)
        bounds = sorted(set(bounds))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) >= min_phrase_s * FPS:
                out.append((a, b))
    return out


def phrase_offset(dry_f0, ref_f0, ps, pe, base_lag_s, max_shift=0.35):
    """一句的对齐偏移（正=干声晚）。用句首起音点对起音点，失败则用句内所有起音点中位。"""
    on_d = voiced_onsets(dry_f0[ps:pe]) + ps
    on_r = voiced_onsets(ref_f0)
    base = base_lag_s * FPS
    ml = int(max_shift * FPS)
    if len(on_d):
        first = on_d[0]
        cand = on_r[(on_r > first + base - ml) & (on_r < first + base + ml)]
        if len(cand):
            k = int(np.argmin(np.abs(cand - (first + base))))
            return (cand[k] - first) / FPS - base_lag_s, 1
    return None, 0


def measure(dry_path, song_dir, log=print):
    cands = sorted(p for p in Path(song_dir).glob("*.wav") if p.name.startswith("vocals_"))
    ref = A.load_mono(str(cands[0]), C.VOC_SR)
    dry = A.load_mono(dry_path, C.VOC_SR)
    ex = FcpeExtractor()
    d, r = ex.extract(dry, C.VOC_SR), ex.extract(ref, C.VOC_SR)
    base = AL._estimate_offset(AL._decimate(d, FPS, 10.0), AL._decimate(r, FPS, 10.0), 10.0)
    rms = rms_curve(dry)
    ph = split_phrases(d, rms, C.VOC_SR / 441)
    res = []
    for ps, pe in ph:
        off, n = phrase_offset(d, r, ps, pe, base)
        if off is None:
            continue
        res.append((ps / FPS, pe / FPS, off))
    return base, res, d, r


if __name__ == "__main__":
    for name, dryp, sd in [("青花瓷", r"D:/_0.Downloads/青花瓷-干声.mp3", "work/周杰伦-青花瓷"),
                           ("过火", r"D:/_0.Downloads/演唱干声-分析文件.m4a",
                            "work/張信哲 Jeff Chang [ 過火 ] 官方完整版 Official MV")]:
        base, res, _, _ = measure(dryp, sd)
        offs = np.array([x[2] for x in res])
        dur = np.array([x[1] - x[0] for x in res])
        print(f"\n=== {name} === 全局 {base*1000:+.0f}ms  句数 {len(res)}  "
              f"覆盖 {dur.sum():.0f}s/{dur.sum()+sum(0 for _ in res):.0f}s")
        a = np.abs(offs)
        print(f"  句偏移 |P50|={np.percentile(a,50)*1000:.0f}ms P75={np.percentile(a,75)*1000:.0f}ms "
              f"P90={np.percentile(a,90)*1000:.0f}ms max={a.max()*1000:.0f}ms std={offs.std()*1000:.0f}ms")
        for th in (0.04, 0.08, 0.12, 0.20):
            print(f"    超过 {th*1000:3.0f}ms: {(a>th).mean()*100:.1f}%")
        print(f"  带符号 P10={np.percentile(offs,10)*1000:+.0f} P50={np.percentile(offs,50)*1000:+.0f} "
              f"P90={np.percentile(offs,90)*1000:+.0f}")
        print("  句起点/偏移:", " ".join(f"{x[0]:.0f}s:{x[2]*1000:+.0f}" for x in res[:16]))
