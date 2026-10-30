"""节奏对齐验收：测「干声 / 成品」相对原唱的**入点偏差**分布。

方法（唯一被严格标定过的一种）：
  1) 全局：两轨浊音起点做成脉冲序列做互相关 → 全局偏移
     标定：原唱平移 ±200ms 共 7 组，误差全 0.0ms
  2) 局部：在全局偏移下做**单调序列对齐**（1D DP，允许跳过），逐点拿偏差
  3) 指标：|偏差| 的 P50 / P90 / max，以及「超过 80ms 的占比」

已证伪、不要再用的方法：
  ✗ onset 能量包络互相关（真值 0ms 测出 -260ms，误差 260~310ms）
  ✗ 10fps / 100fps DTW 漂移（路径贴死带宽，结果是常数）
  ✗ 分段色度互相关（真值 0.7s 只测出 0.3s）
  ✗ 分句音高扫描（值全落 20ms 格点，测的是音高匹配不是节奏）

用法：
  python tools/eval_timing.py 干声 原曲目录 [out/xxx_vocal.wav]
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shadowvox import config as C, audio as A, rhythm as R
from shadowvox.f0 import FcpeExtractor


def stats(dv):
    a = np.abs(dv)
    return {
        "n": len(a),
        "p50": float(np.percentile(a, 50)) * 1000,
        "p90": float(np.percentile(a, 90)) * 1000,
        "max": float(a.max()) * 1000,
        "over80": float((a > 0.08).mean()) * 100,
    }


def evaluate(dry_path, song_dir, placed_path=None, ex=None):
    from pathlib import Path
    ex = ex or FcpeExtractor()
    cands = sorted(p for p in Path(song_dir).glob("*.wav") if p.name.startswith("vocals_"))
    if not cands:
        print(f"找不到原唱轨：{song_dir}")
        return None
    ref = A.load_mono(str(cands[0]), C.VOC_SR)
    rf = ex.extract(ref, C.VOC_SR)

    rows = []
    dry = A.load_mono(dry_path, C.VOC_SR)
    df = ex.extract(dry, C.VOC_SR)
    # 干声未在原曲时间线上 → 用全局偏移对齐后再比
    g = R.onset_offset(df, rf)
    rows.append(("干声(对齐后)", R.onset_alignment(df, rf, g)[1]))

    if placed_path:
        pl = A.load_mono(placed_path, C.VOC_SR)
        pf = ex.extract(pl, C.VOC_SR)
        # 成品已摆到原曲时间线 → 全局偏移按 0 处理
        rows.append(("成品", R.onset_alignment(pf, rf, 0.0)[1]))

    print()
    for name, dv in rows:
        if len(dv) == 0:
            print(f"  {name}: 无有效入点")
            continue
        s = stats(dv)
        print(f"  {name:12s} 配对 {s['n']:3d}  |入点偏差| "
              f"P50={s['p50']:4.0f}ms  P90={s['p90']:4.0f}ms  "
              f"max={s['max']:4.0f}ms  超80ms {s['over80']:4.1f}%")
    if len(rows) == 2 and len(rows[0][1]) and len(rows[1][1]):
        a, b = stats(rows[0][1]), stats(rows[1][1])
        print(f"  → P50 {a['p50']:.0f}→{b['p50']:.0f}ms   P90 {a['p90']:.0f}→{b['p90']:.0f}ms")
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
