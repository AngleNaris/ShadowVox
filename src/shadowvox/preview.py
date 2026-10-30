"""同伴奏 A/B 试听准备；静态响度匹配不改变保存的成品。"""
from pathlib import Path
import numpy as np
import soundfile as sf
from .mixing import measure_loudness


def prepare(before, after):
    paths = [Path(before), Path(after)]
    measures = [measure_loudness(path) for path in paths]
    if not all(np.isfinite(value) for pair in measures for value in pair):
        raise ValueError("音频没有可供试听的有效声音")
    target = min(-19., *(lufs - 1.5 - peak for lufs, peak in measures))
    tracks = []
    sample_rate = None
    for path, (lufs, _) in zip(paths, measures):
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        if sample_rate is not None and sr != sample_rate:
            raise ValueError("对照音频采样率不一致")
        sample_rate = sr
        data *= 10**((target-lufs)/20)
        tracks.append(np.ascontiguousarray(data))
    if tracks[0].shape != tracks[1].shape:
        raise ValueError("对照音频长度不一致")
    return tracks, sample_rate, dict(target_lufs=target,
        source_lufs=[m[0] for m in measures], gains_db=[target-m[0] for m in measures])
