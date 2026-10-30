"""对齐残余：输出人声与原唱人声再跑一次 DTW，统计演唱处偏移（越接近 0 越齐）。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox.audio import load_mono
from shadowvox.full_align import warping_path, singing

def residual(vocal_path, ref_path, sr=44100):
    v, ref = load_mono(vocal_path, sr), load_mono(ref_path, sr)
    a, b = warping_path(v, ref, sr)
    act = singing(v, sr)[np.minimum((a / .1).astype(int), len(v) // round(sr * .1) - 1)]
    off = (b - a)[act]
    rows = []
    for s in range(0, int(len(v) / sr), 10):
        m = (a[act] >= s) & (a[act] < s + 10)
        rows.append('-' if m.sum() < 100 else str(int(round(np.median(off[m]) * 1000))))
    return np.percentile(abs(off) * 1000, [50, 90]), rows

if __name__ == '__main__':
    ref = sys.argv[1]
    for p in sys.argv[2:]:
        (p50, p90), rows = residual(p, ref)
        print(Path(p).name, 'P50 %.0fms P90 %.0fms' % (p50, p90), '| 每10秒:', ' '.join(rows), flush=True)
