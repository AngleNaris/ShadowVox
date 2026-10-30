"""同一参考时间网格、共同浊音帧上的音级误差；不等价于自然度评分。"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shadowvox import config as C, audio as A
from shadowvox.f0 import FcpeExtractor, cent

SR = C.VOC_SR


def err_curve(dry_f0, ref_f0, ref_shift_frames):
    """源网格误差；源帧 j 对应参考帧 j + ref_shift_frames。"""
    idx = np.arange(len(dry_f0)) + ref_shift_frames
    ok = (dry_f0 > 0) & (idx >= 0) & (idx < len(ref_f0))
    out = np.full(len(dry_f0), np.nan)
    m = ok & (ref_f0[np.clip(idx, 0, len(ref_f0) - 1)] > 0)
    if m.any():
        d = np.abs(cent(dry_f0[m]) - cent(ref_f0[np.clip(idx, 0, len(ref_f0) - 1)][m]))
        out[m] = np.minimum(np.mod(d, 1200.0), 1200.0 - np.mod(d, 1200.0))
    return out


def on_reference_grid(f0, offset_seconds, n_frames):
    """offset 正=源晚；参考帧 t 取源帧 t + offset，精度为 10ms。"""
    idx = np.arange(n_frames) + int(round(offset_seconds * C.F0_FRAME_RATE))
    valid = (idx >= 0) & (idx < len(f0))
    out = np.zeros(n_frames, dtype=np.float32)
    out[valid] = f0[idx[valid]]
    return out


def compare(dry_f0, fix_f0, ref_f0, offset_seconds):
    dry = on_reference_grid(dry_f0, offset_seconds, len(ref_f0))
    fix = on_reference_grid(fix_f0, 0, len(ref_f0))
    before, after = err_curve(dry, ref_f0, 0), err_curve(fix, ref_f0, 0)
    common = np.isfinite(before) & np.isfinite(after)
    a, b = before[common], after[common]
    result = {"common_frames": int(common.sum()),
              "dry_valid_frames": int(np.isfinite(before).sum()),
              "fixed_valid_frames": int(np.isfinite(after).sum()),
              "metric": "octave_folded_cents_on_common_reference_grid"}
    if len(a):
        result.update(before_median=float(np.median(a)), after_median=float(np.median(b)),
                      before_mean=float(a.mean()), after_mean=float(b.mean()),
                      improved_ratio=float((b < a).mean()),
                      worsened_over_5c_ratio=float((b > a + 5).mean()))
    return result


def main(dry_path, song_dir, out_vocal, report_path):
    rep = json.loads(open(report_path, encoding="utf-8").read())
    rh = rep.get("rhythm", {})
    off = (rh["onset_offset_before_seconds"] if rh.get("applied")
           else rep["align"]["offset_seconds"])
    print(f"报告 offset = {off:+.4f}s  (干声比原唱晚多少)")

    from pathlib import Path
    d = Path(song_dir)
    cands = sorted(p for p in d.glob("*.wav") if p.name.startswith("vocals_"))
    if len(cands) != 1:
        raise SystemExit(f"需要唯一 vocals_*.wav 参考轨，实际 {len(cands)} 条")
    vstem = str(cands[0])
    ref = A.load_mono(vstem, SR)
    dry = A.load_mono(dry_path, SR)
    fix = A.load_mono(out_vocal, SR)

    ex = FcpeExtractor()
    rf0 = ex.extract(ref, SR)
    print(f"原唱浊音帧 {int((rf0 > 0).sum())}")

    dry_f0 = ex.extract(dry, SR)
    fix_f0 = ex.extract(fix, SR)

    result = compare(dry_f0, fix_f0, rf0, off)
    result["dry_offset_seconds"] = off
    result["note"] = "相同参考时刻比较；变时后不是同一源音素的配对。忽略八度错误，不能衡量自然度。"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
