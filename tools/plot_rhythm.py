"""节奏偏差诊断：把干声/原唱的音高曲线并排画出来，直接看时间错位。

踩过两次坑：
  1) onset 滑窗互相关 → 峰位乱跳、撞搜索边界
  2) 100fps 浊音 DTW → 路径整体贴死带宽边缘，漂移是常数

所以先不急着定量，先把两轨的 F0 曲线按对齐位置叠在一张图上，肉眼确认
「到底有没有局部漂移、量级多大」。输出 PNG。
"""
import sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shadowvox import config as C, audio as A, align as AL
from shadowvox.f0 import FcpeExtractor, cent


def plot_segment(ax, dry_f0, ref_f0, t0, t1, offset, title):
    fps = AL.F0_FPS
    i0, i1 = int(t0 * fps), int(t1 * fps)
    lag = offset * fps
    td = np.arange(i0, min(i1, len(dry_f0))) / fps
    rd = dry_f0[i0:i1]
    # ref 侧下标 = 干声下标 - lag（干声晚 lag 秒）
    ri = (np.arange(i0, min(i1, len(dry_f0))) - lag).astype(int)
    ok = (ri >= 0) & (ri < len(ref_f0))
    ax.plot(td[ok], cent(rd[ok]), lw=1.6, color="#e0803a", label="dry")
    tr = (ri[ok]) / fps
    ax.plot(tr + offset, cent(ref_f0[ri[ok]]), lw=1.6, color="#4a8fd4",
            alpha=0.85, label="ref (aligned)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False

    cases = [("青花瓷", r"D:/_0.Downloads/青花瓷-干声.mp3", "work/周杰伦-青花瓷"),
             ("过火", r"D:/_0.Downloads/演唱干声-分析文件.m4a",
              "work/張信哲 Jeff Chang [ 過火 ] 官方完整版 Official MV")]
    ex = FcpeExtractor()
    fig, axes = plt.subplots(len(cases) * 3, 1, figsize=(16, 4.2 * len(cases) * 3))
    k = 0
    for name, dry_path, song_dir in cases:
        cands = sorted(p for p in Path(song_dir).glob("*.wav") if p.name.startswith("vocals_"))
        ref = A.load_mono(str(cands[0]), C.VOC_SR)
        dry = A.load_mono(dry_path, C.VOC_SR)
        d, r = ex.extract(dry, C.VOC_SR), ex.extract(ref, C.VOC_SR)
        off = AL._estimate_offset(AL._decimate(d, AL.F0_FPS, 10.0),
                                  AL._decimate(r, AL.F0_FPS, 10.0), 10.0)
        print(f"{name}: offset={off*1000:+.0f}ms  dry={len(d)/AL.F0_FPS:.0f}s ref={len(r)/AL.F0_FPS:.0f}s")
        dur = min(len(d), len(r)) / AL.F0_FPS
        for frac in (0.25, 0.5, 0.75):
            t0 = dur * frac
            plot_segment(axes[k], d, r, t0, t0 + 8, off,
                         f"{name} @ {t0:.0f}s (offset {off*1000:+.0f}ms)")
            k += 1
    plt.tight_layout()
    out = "out/rhythm_diag.png"
    plt.savefig(out, dpi=110)
    print("→", out)


if __name__ == "__main__":
    main()
