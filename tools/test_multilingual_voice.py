"""已有合成中英文语音的端到端回归：双方各漏一句，后续句子仍需正确对齐。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox import full_align, lyric_anchors

SR = 22050


def assemble(folder, side, excluded, continuous=False):
    parts, intervals, position = [], {}, 0
    for index in range(8):
        if index in excluded:
            continue
        x, sr = sf.read(folder / "tts" / f"{side}_{index:02d}.wav", dtype="float32")
        assert sr == SR and x.ndim == 1
        if continuous:
            active = np.flatnonzero(np.abs(x) > .005)
            x = x[max(0, active[0] - 220):min(len(x), active[-1] + 220)]
        start = position / SR
        parts.append(x)
        position += len(x)
        intervals[index] = (start, position / SR)
        silence = np.zeros(0 if continuous else round(SR * .7), np.float32)
        parts.append(silence)
        position += len(silence)
    return np.concatenate(parts), intervals


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/multilingual_research"
    ref, ref_intervals = assemble(folder, "reference", {6})
    dry, dry_intervals = assemble(folder, "dry", {3})
    observed, paths, pieces = [], [], []
    original_anchors = lyric_anchors.find_anchors
    original_path = full_align._warping_path
    original_plan = full_align.plan_segments

    def capture_anchors(*args, **kwargs):
        anchors = original_anchors(*args, **kwargs)
        observed.extend(anchors)
        return anchors

    def capture_path(*args, **kwargs):
        path = original_path(*args, **kwargs)
        paths.append(path)
        return path

    def capture_plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        pieces.extend(result[0])
        return result

    lyric_anchors.find_anchors = capture_anchors
    full_align._warping_path = capture_path
    full_align.plan_segments = capture_plan
    begin = time.monotonic()
    _, stats = full_align.align(dry, ref, SR)

    def phrase_at(t, intervals):
        return next((i for i, (a, b) in intervals.items() if a - .15 <= t <= b + .15), None)

    pairs = [(round(a, 2), round(b, 2), phrase_at(a, dry_intervals), phrase_at(b, ref_intervals))
             for a, b in observed if a > 0 and b > 0]
    valid = [(a, b) for _, _, a, b in pairs if a is not None and a == b]
    coverage = sorted({a for a, _ in valid})
    assert len(pairs) >= 3 and len(valid) / len(pairs) >= .8 and len(coverage) >= 4, pairs
    assert stats["alignment_basis"] == "lyrics+audio" and len(paths) >= 2, stats
    dry_t, ref_t = paths[-1]
    locations = {}
    for index, (start, end) in dry_intervals.items():
        if index not in ref_intervals:
            continue
        target = float(np.interp((start + end) / 2, dry_t, ref_t))
        locations[index] = phrase_at(target, ref_intervals)
    assert all(index == mapped for index, mapped in locations.items()), locations
    missing_start, missing_end = dry_intervals[6]
    protected = sum(max(0., min(missing_end, p["source"][-1]) - max(missing_start, p["source"][0]))
                    for p in pieces if p["protected"])
    assert protected >= .5 * (missing_end - missing_start), (protected, stats)
    assert stats["reference_gaps"] >= 1, stats
    # 将相邻句子的静音裁掉，检验没有明显停顿时的有限跨语种匹配。
    dry_continuous, dry_continuous_intervals = assemble(folder, "dry", {3}, continuous=True)
    ref_continuous, ref_continuous_intervals = assemble(folder, "reference", {6}, continuous=True)
    continuous = original_anchors(dry_continuous, ref_continuous, SR)
    continuous_pairs = [(phrase_at(a, dry_continuous_intervals), phrase_at(b, ref_continuous_intervals))
                        for a, b in continuous]
    continuous_valid = [(a, b) for a, b in continuous_pairs if a is not None and a == b]
    continuous_coverage = sorted({a for a, _ in continuous_valid})
    assert len(continuous_coverage) >= 3 and len(continuous_valid) == len(continuous_pairs), continuous_pairs
    summary = {"anchors": len(pairs), "same_phrase": len(valid), "phrase_coverage": coverage,
               "locations": locations, "missing_source_protected_seconds": round(protected, 2),
               "protected_seconds": stats["protected_seconds"], "continuous_phrase_coverage": continuous_coverage,
               "elapsed_seconds": round(time.monotonic() - begin, 1)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
