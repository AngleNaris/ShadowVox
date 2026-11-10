"""保守对齐：原声顺序、完整性和连续性优先于贴齐参考。

Sync Toolbox 只提出候选位置。可信区域的位移在边界回到零，Rubber Band R3
只处理这些区域；其余波形逐采样保留，不切句、插空或移动后句来追赶路径。
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import distance_transform_edt, uniform_filter1d
from scipy.signal import resample_poly

from .audio import subprocess_options
from .natural import runs
from .rubberband import executable

FEATURE_SR, FEATURE_RATE = 22050, 50
KEY_SECONDS = .1
MAX_LOCAL_SHIFT = .15
ANCHOR_REACH_SECONDS = 4.
FADE_SECONDS = .03
GAP_SECONDS = .18


def features(audio, sr):
    from .vendor.synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from .vendor.synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from .vendor.synctoolbox.feature.pitch import audio_to_pitch_features
    from .vendor.synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    y = resample_poly(audio, FEATURE_SR, sr).astype(np.float64)
    chroma = quantize_chroma(pitch_to_chroma(audio_to_pitch_features(y, Fs=FEATURE_SR, feature_rate=FEATURE_RATE)))
    onsets = pitch_onset_features_to_DLNCO(audio_to_pitch_onset_features(y, Fs=FEATURE_SR),
                                           feature_sequence_length=chroma.shape[1], feature_rate=FEATURE_RATE)
    return chroma, onsets


def _warping_path(c1, o1, c2, o2, anchors=None):
    from .vendor.synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw, sync_via_mrmsdtw_with_anchors
    from .vendor.synctoolbox.dtw.utils import make_path_strictly_monotonic
    kwargs = dict(f_chroma1=c1, f_onset1=o1, f_chroma2=c2, f_onset2=o2,
                  input_feature_rate=FEATURE_RATE, alpha=.5)
    if anchors:
        path = sync_via_mrmsdtw_with_anchors(**kwargs, anchor_pairs=list(anchors))
    else:
        path = sync_via_mrmsdtw(**kwargs)
    path = make_path_strictly_monotonic(path)
    return path[0] / FEATURE_RATE, path[1] / FEATURE_RATE


def warping_path(dry, reference, sr):
    """返回候选单调路径；调用方仍须核验，不可直接作为音频搬移指令。"""
    c1, o1 = features(dry, sr)
    c2, o2 = features(reference, sr)
    return _warping_path(c1, o1, c2, o2)


def singing(audio, sr):
    hop = round(sr * KEY_SECONDS)
    count = max(1, int(np.ceil(len(audio) / hop)))
    padded = np.pad(audio, (0, count * hop - len(audio)))
    rms = np.sqrt(np.mean(np.square(padded.reshape(count, hop), dtype=np.float64), axis=1))
    active = rms > .1 * np.percentile(rms, 90)
    return active | np.r_[active[1:], False] | np.r_[False, active[:-1]]


def stretch(x, source, target, sr):
    """R3 单次变时；拒绝会反转、重复或跳过源时间的映射。"""
    frames = np.rint(np.c_[source * sr, target * sr]).astype(np.int64)
    if (len(frames) < 2 or np.any(np.diff(frames, axis=0) <= 0) or
            np.any(frames[0] != 0)):
        raise ValueError("变时映射必须从零开始且两侧严格递增")
    with tempfile.TemporaryDirectory(prefix="shadowvox-align-") as tmp:
        tmp = Path(tmp)
        sf.write(tmp / "input.wav", x, sr, subtype="FLOAT")
        (tmp / "timemap.txt").write_text("\n".join(f"{a} {b}" for a, b in frames), encoding="ascii")
        ratio = frames[-1, 1] / frames[-1, 0]
        p = subprocess.run([executable(), "-3", "-t", f"{ratio:.8f}", "--timemap", "timemap.txt", "-q",
                            "input.wav", "output.wav"], cwd=tmp, capture_output=True,
                           **subprocess_options())
        if p.returncode:
            raise RuntimeError(p.stderr.decode(errors="replace"))
        return sf.read(tmp / "output.wav", dtype="float32")[0]


def _ordered(left, right):
    return (len(left) >= 2 and len(left) == len(right) and
            np.isfinite(left).all() and np.isfinite(right).all() and
            np.all(np.diff(left) > 0) and np.all(np.diff(right) > 0))


def _global_offset(anchors, dry_t, ref_t):
    """至少三处相隔的乐句与声学路径共同证实，才补偿整段录音延迟。"""
    if len(anchors) < 4:
        return 0., False
    a, b = np.asarray(anchors, dtype=float).T
    if not _ordered(a, b) or not _ordered(dry_t, ref_t):
        return 0., False
    delta = b - a
    bins = (a / 8).astype(int)
    medians = np.array([np.median(delta[bins == k]) for k in np.unique(bins)])
    centre = float(np.median(medians))
    good = np.abs(delta - centre) <= .2
    # 小偏差留给可信局部修正；存在两种时标解释时保持原位。
    if abs(centre) <= .15 or (abs(centre) > .5 and np.count_nonzero(np.abs(medians) <= .2) >= 2):
        return 0., False
    if (good.sum() < 4 or len(np.unique(bins[good])) < 3 or np.ptp(a[good]) < 12 or
            np.mean(np.abs(medians - centre) <= .2) < .75):
        return 0., False
    audio_delta = np.interp(a[good], dry_t, ref_t) - a[good]
    if np.percentile(np.abs(audio_delta - delta[good]), 75) > .15:
        return 0., False
    return float(np.median(delta[good])), True


def _local_anchors(anchors, offset):
    if anchors is None or len(anchors) == 0:
        return []
    a, b = np.asarray(anchors, dtype=float).T
    if not _ordered(a, b):
        return []                # 不通过排序“修复”冲突歌词
    delta = b - a
    near = np.abs(delta - offset) <= .35
    result = []
    for i in np.flatnonzero(near):
        support = near & (np.abs(a - a[i]) <= 8.) & (np.abs(delta - delta[i]) <= .2)
        if support.sum() >= 2:
            result.append((float(a[i]), float(b[i])))
    return result


def _lyric_support(grid, anchors, offset):
    """仅在相邻、一致锚点之间使用字句证据，不向两端外推。"""
    trusted = np.zeros(len(grid), bool)
    expected = np.full(len(grid), offset)
    for (a, b), (c, d) in zip(anchors, anchors[1:]):
        if c - a <= 2 * ANCHOR_REACH_SECONDS and abs((d - c) - (b - a)) <= .2:
            inside = (grid >= a) & (grid <= c)
            trusted |= inside
            expected[inside] = np.interp(grid[inside], [a, c], [b - a, d - c])
    return trusted, expected


def _gap_groups(dry_t, ref_t, source_gap):
    dx, dy = np.diff(dry_t), np.diff(ref_t)
    candidates = np.flatnonzero((dx - dy if source_gap else dy - dx) > GAP_SECONDS)
    groups = []
    for i in candidates:
        if groups and dry_t[i] - dry_t[groups[-1][-1] + 1] <= .6 and ref_t[i] - ref_t[groups[-1][-1] + 1] <= .6:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [g for g in groups if sum((dx[i] - dy[i] if source_gap else dy[i] - dx[i]) for i in g) >= .3]


def plan_segments(dry_t, ref_t, dry_seconds, ref_seconds, voiced, anchors=None, offset=0.,
                  acoustic_trusted=None, strength=1., max_rate=.10):
    """完整覆盖原声、相邻端点相接；保护区的位移与速度严格保持基线。

    offset 是已核实的整段位移。局部证据不足或超过 150ms 时不执行；
    调整边界回到基线，因此不会把错误传播到后面的字句。
    """
    source = np.r_[np.arange(0., dry_seconds, KEY_SECONDS), dry_seconds]
    sung = voiced[np.minimum(np.arange(len(source) - 1), len(voiced) - 1)]
    allowed = np.zeros(len(sung), bool)
    proposal = source + offset
    ordered = _ordered(dry_t, ref_t)
    if ordered and acoustic_trusted is not None:
        proposal = np.interp(source, dry_t, ref_t)
        allowed = sung & acoustic_trusted[np.minimum(np.arange(len(sung)), len(acoustic_trusted) - 1)]
        if anchors is not None:
            valid = _local_anchors(anchors, offset)
            words_ok, expected = _lyric_support(source, valid, offset)
            allowed &= words_ok[:-1] & words_ok[1:]
            allowed &= np.abs(proposal[:-1] - source[:-1] - expected[:-1]) <= .15
        difference = proposal - source - offset
        allowed &= np.abs(difference[:-1]) <= MAX_LOCAL_SHIFT + 1e-8
        allowed &= np.abs(difference[1:]) <= MAX_LOCAL_SHIFT + 1e-8
        speed = np.diff(proposal) / np.diff(source)
        allowed &= (speed >= .75) & (speed <= 1.25)
    for lo, hi in runs(allowed):
        if source[hi] - source[lo] < .6:
            allowed[lo:hi] = False

    # 保护单元两端必须为零，不能仅保持它内部的速度。
    movable = np.r_[False, allowed[:-1] & allowed[1:], False]
    rate = float(np.clip(max_rate, 0., .5))
    ceiling = np.minimum(MAX_LOCAL_SHIFT, distance_transform_edt(movable) * KEY_SECONDS * rate)
    delta = np.clip(proposal - source - offset, -ceiling, ceiling)
    delta[~movable] = 0.
    for i in range(1, len(delta)):
        step = rate * (source[i] - source[i - 1])
        delta[i] = np.clip(delta[i], delta[i - 1] - step, delta[i - 1] + step)
    for i in range(len(delta) - 2, -1, -1):
        step = rate * (source[i + 1] - source[i])
        delta[i] = np.clip(delta[i], delta[i + 1] - step, delta[i + 1] + step)
    delta *= float(np.clip(strength, 0., 1.))
    # 提前整轨时延后参考，完整保留开头，而不是剪掉源波形。
    reference_shift = max(0., -offset)
    target = source + offset + reference_shift + delta
    cuts = np.r_[0, np.flatnonzero(np.diff(allowed.astype(np.int8))) + 1, len(allowed)]
    pieces = [{"source": source[a:b + 1], "target": target[a:b + 1], "protected": not bool(allowed[a])}
              for a, b in zip(cuts, cuts[1:])]
    protected = [[float(p["target"][0]), float(p["target"][-1])] for p in pieces if p["protected"]]
    rates = np.diff(target) / np.diff(source)
    stats = {
        "protected_intervals_seconds": protected,
        "protected_seconds": round(float(np.sum(np.diff(source)[sung & ~allowed])), 3),
        "active_intervals_seconds": [[float(p["target"][0]), float(p["target"][-1])]
                                     for p in pieces if not p["protected"]],
        "reference_gaps": len(_gap_groups(dry_t, ref_t, False)) if ordered else 0,
        "source_gaps": len(_gap_groups(dry_t, ref_t, True)) if ordered else 0,
        "reference_shift_seconds": float(reference_shift),
        "extra_duration_seconds": round(max(0., target[-1] - ref_seconds - reference_shift), 3),
        "max_rate": rate, "max_local_shift_seconds": MAX_LOCAL_SHIFT,
        "max_applied_shift_ms": round(float(np.max(np.abs(delta))) * 1000, 3),
        "rate_min": round(float(rates.min()), 6), "rate_max": round(float(rates.max()), 6),
        "rate_clipped_ratio": round(float(np.mean(np.abs(rates - 1) >= rate - 1e-7)), 3) if rate else 0.,
        "order_preserved": True, "continuous_source_coverage": True,
        "applied": bool(np.max(np.abs(delta)) > 1e-4 or abs(offset) > 1e-4),
    }
    return pieces, stats


def _correspondence_support(c1, c2, dry_t, ref_t, voiced, anchors, offset=0.):
    """对应关系的证据；不包含变时幅度、边界或渲染是否成功。"""
    grid = np.arange(len(voiced)) * KEY_SECONDS
    ref_grid = np.interp(grid, dry_t, ref_t)
    left = np.clip(np.rint(grid * FEATURE_RATE).astype(int), 0, c1.shape[1] - 1)
    right = np.clip(np.rint(ref_grid * FEATURE_RATE).astype(int), 0, c2.shape[1] - 1)
    a, b = c1[:, left], c2[:, right]
    denom = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    similarity = np.divide(np.sum(a * b, axis=0), denom, out=np.zeros_like(denom), where=denom > 1e-8)
    similarity = uniform_filter1d(similarity, size=9, mode="nearest")
    speed = np.interp(grid + .5, dry_t, ref_t) - np.interp(grid - .5, dry_t, ref_t)
    valid = voiced & (similarity >= .6) & (speed >= .8) & (speed <= 1.2)
    if anchors is not None:
        words_ok, expected = _lyric_support(grid, anchors, offset)
        valid &= words_ok & (np.abs(ref_grid - grid - expected) <= .15)
    return valid


def _acoustic_support(c1, c2, dry_t, ref_t, voiced, anchors, offset=0.):
    valid = _correspondence_support(c1, c2, dry_t, ref_t, voiced, anchors, offset)
    grid = np.arange(len(voiced)) * KEY_SECONDS
    return valid & (np.abs(np.interp(grid, dry_t, ref_t) - grid - offset) <= MAX_LOCAL_SHIFT)


def pitch_mapping(pieces, plan, frame_count, frame_rate=100., start_seconds=0.):
    """实际输出→原声→参考；变时回退只改变第一步，不否决对应证据。"""
    source = np.concatenate([p['source'][:-1] for p in pieces] + [pieces[-1]['source'][-1:]])
    output = np.concatenate([p['target'][:-1] for p in pieces] + [pieces[-1]['target'][-1:]])
    times = start_seconds + np.arange(frame_count) / frame_rate
    source_times = np.interp(times, output, source)
    reference_shift = pieces[0].get('reference_shift', max(0., -plan['offset']))
    pairs = plan.get('pitch_pairs')
    if pairs is not None:
        if not len(pairs):
            return np.zeros(frame_count), np.zeros(frame_count, dtype=bool)
        starts, ends, ref_starts, ref_ends = pairs.T
        index = np.searchsorted(starts,source_times,side='right')-1
        valid = index >= 0
        index = np.maximum(index,0)
        valid &= source_times < ends[index]
        progress = np.clip((source_times-starts[index])/(ends[index]-starts[index]),0.,1.)
        reference_times = ref_starts[index] + progress*(ref_ends[index]-ref_starts[index]) + reference_shift
    else:
        reference_times = np.interp(source_times, plan['dry_t'], plan['ref_t']) + reference_shift
        evidence = plan.get('correspondence_trusted', plan['acoustic_trusted'])
        cells = np.floor(source_times / KEY_SECONDS + 1e-8).astype(int)
        valid = evidence[np.clip(cells, 0, len(evidence)-1)].copy()
        valid &= (source_times >= plan['dry_t'][0]) & (source_times <= plan['dry_t'][-1])
    valid &= (times >= output[0]) & (times < output[-1])
    valid &= (reference_times >= reference_shift) & (reference_times < plan['ref_seconds'] + reference_shift)
    return reference_times, valid


def render_segments(dry, pieces, sr, start=0, end=None):
    """整曲与草稿共用入口；中文模式连续变时，旧分段模式的保护片段直接旁路。"""
    end = round(pieces[-1]["target"][-1] * sr) if end is None else end
    if pieces[0].get('continuous'):
        source = np.concatenate([p['source'][:-1] for p in pieces]+[pieces[-1]['source'][-1:]])
        target = np.concatenate([p['target'][:-1] for p in pieces]+[pieces[-1]['target'][-1:]])
        origin = round(target[0]*sr)
        target -= target[0]
        rendered = dry.copy() if np.max(abs(target-source)) < 1e-7 else stretch(dry,source,target,sr)
        if abs(len(rendered)-len(dry)) > 2:
            raise RuntimeError('中文对齐输出长度不符，已停止导出')
        rendered = np.pad(rendered,(0,max(0,len(dry)-len(rendered))),mode='edge')[:len(dry)]
        out = np.zeros(end-start,dtype=np.float32)
        lo,hi = max(origin,start),min(end,origin+len(rendered))
        if hi > lo: out[lo-start:hi-start] = rendered[lo-origin:hi-origin]
        return out, [(float(p['target'][0]),float(p['target'][-1])) for p in pieces if p['protected']]
    out = np.zeros(end - start, dtype=np.float32)
    protected = []
    for piece in pieces:
        src, dst = piece["source"], piece["target"]
        lo, hi = max(start / sr, dst[0]), min(end / sr, dst[-1])
        if hi <= lo:
            continue
        # 可信片段整段计算，保证草稿与整曲的变时及边界判定一致。
        lo, hi = ((max(dst[0], lo - .25), min(dst[-1], hi + .25)) if piece['protected']
                  else (dst[0], dst[-1]))
        src_lo, src_hi = np.interp([lo, hi], dst, src)
        inside = (src > src_lo + 1/sr) & (src < src_hi - 1/sr)
        local_src = np.r_[src_lo, src[inside], src_hi]
        local_dst = np.r_[lo, dst[inside], hi]
        x = dry[round(src_lo * sr):round(src_hi * sr)]
        want = round(hi * sr) - round(lo * sr)
        unchanged = np.max(np.abs((local_dst - lo) - (local_src - src_lo))) < 1e-7
        if piece["protected"] or unchanged:
            y = x.copy()
        else:
            y = stretch(x, local_src - src_lo, local_dst - lo, sr)
            if abs(len(y) - want) > 2:
                raise RuntimeError("变时输出长度不符，已停止导出以避免插入静音或截断人声")
            fade = min(round(sr * FADE_SECONDS), len(y), len(x))
            # 相位相反时交叉淡化会压低连续人声，整段回退到原位原声。
            if fade and any(np.dot(a.astype(float), b.astype(float)) < 0
                            for a, b in ((x[:fade], y[:fade]), (x[-fade:], y[-fade:]))):
                y = x.copy()
                piece['target'] = src + (dst[0] - src[0])
                piece['protected'] = True
                piece['render_fallback'] = 'opposed_boundary_phase'
                fade = 0
            weight = np.linspace(0., 1., fade, dtype=np.float32)
            if lo == dst[0] and fade:
                y[:fade] = x[:fade] * (1 - weight) + y[:fade] * weight
            if hi == dst[-1] and fade:
                y[-fade:] = x[-fade:] * weight + y[-fade:] * (1 - weight)
        y = np.pad(y[:want], (0, max(0, want - len(y))), mode="edge")
        at = round(lo * sr) - start
        a, b = max(0, -at), min(len(y), end - start - at)
        if b > a:
            out[at + a:at + b] = y[a:b]
        if piece["protected"]:
            protected.append((lo, hi))
    return out, protected


def align(dry, reference, sr, use_lyrics=True, log=print, strength=1., capture=None, max_rate=.10,
          backend='legacy', offset_hint=0.):
    if backend == 'hubertfa_zh':
        from .hubert_alignment import align as chinese_align
        return chinese_align(dry,reference,sr,log,strength,capture,max_rate,offset_hint)
    c1, o1 = features(dry, sr)
    c2, o2 = features(reference, sr)
    dry_t, ref_t = _warping_path(c1, o1, c2, o2)
    voiced = singing(dry, sr)
    anchors = None
    candidates = []
    lyric_error = None
    offset, global_reliable = 0., False
    if use_lyrics:
        from .lyric_anchors import find_anchors
        log("[节奏] 检查顺序一致的字句与声学证据")
        try:
            candidates = find_anchors(dry, reference, sr, 0.)
            candidates = [(a, b) for a, b in candidates if 0 < a < len(dry)/sr - .1 and
                          0 < b < len(reference)/sr - .1]
        except Exception as exc:
            lyric_error = str(exc)
            log(f"[节奏] 字句识别不可用，保留原位置：{exc}")
        offset, global_reliable = _global_offset(candidates, dry_t, ref_t)
        offset = round(offset * sr) / sr
        anchors = _local_anchors(candidates, offset)
        if len(anchors) >= 3:
            dry_t, ref_t = _warping_path(c1, o1, c2, o2, anchors)
        else:
            anchors = []
        log(f"[节奏] 接受 {len(anchors)} 个锚点，拒绝 {len(candidates)-len(anchors)} 个；不外推未匹配片段")
    acoustic_trusted = _acoustic_support(c1, c2, dry_t, ref_t, voiced, anchors, offset)
    plan = dict(dry_t=dry_t, ref_t=ref_t, dry_seconds=len(dry)/sr, ref_seconds=len(reference)/sr,
                voiced=voiced, anchors=anchors, offset=offset, acoustic_trusted=acoustic_trusted)
    pieces, stats = plan_segments(**plan, strength=strength, max_rate=max_rate)
    if capture is not None:
        capture['rhythm_plan'] = dict(plan, correspondence_trusted=_correspondence_support(
            c1, c2, dry_t, ref_t, voiced, anchors, offset))
    length = max(len(reference) + round(stats['reference_shift_seconds'] * sr),
                 round(pieces[-1]['target'][-1] * sr))
    out, _ = render_segments(dry, pieces, sr, end=length)
    if capture is not None:
        capture['rhythm_pieces'] = pieces
    # 渲染回退也必须保护后续音高修正，报告只描述实际采用的映射。
    stats['render_fallbacks'] = sum('render_fallback' in p for p in pieces)
    if stats['render_fallbacks']:
        stats['protected_intervals_seconds'] = [[float(p['target'][0]), float(p['target'][-1])]
                                                for p in pieces if p['protected']]
        stats['active_intervals_seconds'] = [[float(p['target'][0]), float(p['target'][-1])]
                                             for p in pieces if not p['protected']]
        stats['protected_seconds'] = round(sum(float(np.sum(np.diff(p['source']) *
            voiced[np.minimum(np.rint(p['source'][:-1]/KEY_SECONDS).astype(int), len(voiced)-1)]))
            for p in pieces if p['protected']), 3)
        rates = np.concatenate([np.diff(p['target']) / np.diff(p['source']) for p in pieces])
        shift = max(float(np.max(np.abs(p['target']-p['source']-max(0., offset)))) for p in pieces)
        stats.update(rate_min=round(float(rates.min()), 6), rate_max=round(float(rates.max()), 6),
                     max_applied_shift_ms=round(shift*1000, 3), applied=bool(shift > 1e-4 or abs(offset) > 1e-4))
        stats['rate_clipped_ratio'] = (round(float(np.mean(np.abs(rates-1) >= stats['max_rate']-1e-7)), 3)
                                       if stats['max_rate'] else 0.)
    stats.update(algorithm="conservative-lyrics+rubberband-r3-v2", runs=len(pieces),
                 lyric_anchors=len(anchors or []), rejected_lyric_anchors=len(candidates)-len(anchors or []),
                 alignment_basis="lyrics+audio" if anchors else ("audio" if anchors is None else "preserved"),
                 median_offset_seconds=float(offset), placement_offset_seconds=0.,
                 global_offset_reliable=global_reliable)
    if lyric_error:
        stats['lyric_error'] = lyric_error
    return out, stats
