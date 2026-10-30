"""整句对齐已知答案测试：把一段真实干声按已知曲线变速、前面加长前奏、中间换气处插入长间奏，
当作“原唱”，再用 full_align 对齐回去，逐 2 秒窗口测输出与“原唱”的残余错位。

覆盖本次真实素材暴露的问题：空白处需要极端伸缩（干声 0.5 秒换气 ↔ 原唱 6 秒间奏）时，
Rubber Band 单次变速跟不上，会留下秒级的整体错位。
用法：python tools/test_full_align.py [干声路径]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadowvox.audio import load_mono
from shadowvox.full_align import align, singing

SR = 44100
DRY = r"D:\_0.Downloads\青花瓷-干声.mp3"


def warp(x, sr):
    """按已知曲线（0.8~1.25 倍缓慢起伏）变速，返回变速后音频。"""
    from shadowvox.full_align import stretch
    src = np.arange(0, len(x) / sr, .1)
    rate = 2 ** (np.sin(src / 4) * np.log2(1.25))
    dst = np.r_[0, np.cumsum(rate[:-1] * .1)]
    return stretch(x, src, dst, sr)


def envelope(x, sr):
    hop = sr // 1000
    n = len(x) // hop
    return np.sqrt(np.mean(np.square(x[:n * hop].reshape(n, hop)), axis=1))


def lags(out, ref, sr, active):
    """每 2 秒窗口（步长 1 秒）包络互相关，±400ms 内最佳错位（ms）。"""
    a, b = envelope(out, sr), envelope(ref, sr)
    result = []
    for s in range(400, min(len(a), len(b)) - 2400, 1000):
        if not active[s // 100:(s + 2000) // 100].mean() > .6:
            continue
        w = b[s:s + 2000] - b[s:s + 2000].mean()
        best = max(range(-400, 401, 2), key=lambda L: np.dot(w, a[s + L:s + L + 2000] - a[s + L:s + L + 2000].mean()))
        result.append(best)
    return np.abs(result)


def main():
    dry = load_mono(sys.argv[1] if len(sys.argv) > 1 else DRY, SR)[SR * 30:SR * 90]
    cell = round(SR * .1)
    rms = np.sqrt(np.mean(np.square(dry[:len(dry) // cell * cell].reshape(-1, cell)), axis=1))
    cut = (200 + int(np.argmin(rms[200:400]))) * cell          # 20~40 秒内最安静处当作换气
    dry = np.r_[dry[:cut], np.zeros(SR // 2, np.float32), dry[cut:]]
    ref = warp(dry, SR)
    gap = round(cut / len(dry) * len(ref))
    ref = np.r_[np.zeros(SR * 10, np.float32), ref[:gap], np.zeros(SR * 6, np.float32), ref[gap:]].astype(np.float32)
    out, stats = align(dry, ref, SR)
    lag = lags(out, ref, SR, singing(ref, SR))
    p50, p90 = np.percentile(lag, [50, 90])
    print("已知答案 |残余错位| P50 %.0fms P90 %.0fms 最大 %.0fms 窗口 %d" % (p50, p90, lag.max(), len(lag)), stats)
    assert p90 < 80 and lag.max() < 200, "整句对齐残余错位过大"
    print("ok")


if __name__ == "__main__":
    main()
