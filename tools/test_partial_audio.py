"""用真实干声构造可验证的双向漏唱；不会修改源音频。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadowvox.audio import load_mono
from shadowvox.full_align import align, singing
from test_full_align import lags


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: python tools/test_partial_audio.py 干声路径")
    sr = 44100
    full = load_mono(sys.argv[1], sr)
    base = full[80 * sr:110 * sr]
    if len(base) != 30 * sr:
        raise ValueError("测试素材需至少 110 秒")
    cut = np.r_[base[:10 * sr], base[12 * sr:]]

    # 演唱少两秒：缺口之后应立即恢复，而不是把后续词越拉越慢。
    corrected, stats = align(cut, base, sr, use_lyrics=False, log=lambda _: None)
    active = singing(base, sr)
    active[80:140] = False
    errors = lags(corrected, base, sr, active)
    p90 = float(np.percentile(errors, 90))
    print("dry missing: P90 %.0fms, reference gaps %d" % (p90, stats["reference_gaps"]))
    assert p90 < 80 and stats["reference_gaps"] > 0

    # 参考少两秒：多出的演唱按原速度保留，不能把后续部分硬挤进去。
    corrected, stats = align(base, cut, sr, use_lyrics=False, log=lambda _: None)
    print("reference missing: protected %.2fs, extra %.2fs" %
          (stats["protected_seconds"], stats["extra_duration_seconds"]))
    assert stats["protected_seconds"] >= 1.5
    # 足够的换气空白可以吸收时长差，不要求成品一定比原曲更长。
    assert stats["source_gaps"] > 0 and len(corrected) >= len(cut)
    assert np.isfinite(corrected).all()


if __name__ == "__main__":
    main()
