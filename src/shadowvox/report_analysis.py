"""报告用的实际结果分析；不参与修音、变时或混音决策。"""
import numpy as np
import soundfile as sf

from . import audio, config as C, rhythm
from .natural import runs

CENTERS = (40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000)
TONE_BANDS = (('声体', 120, 450), ('清晰', 1000, 4000), ('空气感', 8000, 12000))


def _numbers(values, digits=3):
    return [round(float(v), digits) if np.isfinite(v) else None for v in values]


def _placed_frames(values, move, length):
    """内容 move>0 表示延后；输出时刻 t 读取源时刻 t-move。"""
    indices = np.arange(length)-round(move*C.F0_FRAME_RATE)
    valid = (indices >= 0) & (indices < len(values))
    result = np.zeros(length, dtype=np.float64)
    result[valid] = np.asarray(values)[indices[valid]]
    return result


def _cents(values):
    return 1200*np.log2(np.maximum(values, 1e-8)/440)+6900


def _frame_intervals(mask):
    return [[float(a/C.F0_FRAME_RATE), float(b/C.F0_FRAME_RATE)] for a,b in runs(mask)]


def pitch_result(before, after, reference):
    n = min(len(before), len(after), len(reference))
    before, after, reference = (np.asarray(x[:n]) for x in (before, after, reference))
    input_voiced = before > 0
    with_reference = input_voiced & (reference > 0)
    common = with_reference & (after > 0)
    missing = input_voiced & (after <= 0)
    coverage = {
        'input_voiced_frames': int(np.sum(input_voiced)),
        'input_reference_voiced_frames': int(np.sum(with_reference)),
        'output_f0_missing_frames': int(np.sum(missing)),
        'output_f0_missing_on_reference_frames': int(np.sum(with_reference & missing)),
        'output_f0_missing_intervals_seconds': _frame_intervals(missing),
    }
    if np.sum(common) < 100:
        return {'available': False, 'reason': '共同有效音高不足', 'common_frames': int(np.sum(common)),
                'coverage': coverage}
    bc, ac, rc = (_cents(x) for x in (before, after, reference))
    # 保留演唱者音区；同一参考、同一有效帧、同一音级误差口径。
    bd, ad = ((x-rc+600) % 1200-600 for x in (bc, ac))
    target = rc+1200*np.rint((bc-rc)/1200)
    window = min(n, round(12*C.F0_FRAME_RATE))
    starts = np.arange(0, max(1, n-window+1), round(2*C.F0_FRAME_RATE))
    start = int(max(starts, key=lambda s: np.sum(common[s:s+window])))
    end = start+window
    indices = np.unique(np.linspace(start, end-1, min(360, window)).astype(int))
    valid = common[indices]
    def curve(values):
        return _numbers(np.where(valid, values[indices], np.nan))
    b, a = np.abs(bd[common]), np.abs(ad[common])
    return {'available': True, 'common_frames': int(np.sum(common)), 'coverage': coverage,
            'before_median': round(float(np.median(b)), 1), 'after_median': round(float(np.median(a)), 1),
            'before_p90': round(float(np.percentile(b, 90)), 1),
            'after_p90': round(float(np.percentile(a, 90)), 1),
            'metric': 'octave_folded_cents_on_common_frames',
            'window': {'start': start/C.F0_FRAME_RATE, 'end': end/C.F0_FRAME_RATE,
                'times': _numbers(indices/C.F0_FRAME_RATE, 2),
                'reference': curve(target/100), 'before': curve((target+bd)/100),
                'after': curve((target+ad)/100),
                'before_deviation': curve(bd), 'after_deviation': curve(ad)}}


def timing_result(before, after, reference):
    """前后分别匹配实际浊音入点，再只保留共同的原唱入点；不把计划等同实测。"""
    rows, counts = [], []
    for source in (before, after):
        times, deviation = rhythm.onset_alignment(source, reference, 0.)
        rows.append({round((t+d)*C.F0_FRAME_RATE): (float(t), float(-d*1000))
                     for t, d in zip(times, deviation)})
        counts.append(len(times))
    shared = sorted(rows[0].keys() & rows[1].keys())
    ref_onsets = rhythm.voiced_onsets(reference)
    result = {'available': len(shared) >= 3, 'paired_onsets': len(shared),
              'before_matched': counts[0], 'after_matched': counts[1],
              'reference_onsets': len(ref_onsets),
              'before_unmatched_reference': len(ref_onsets)-counts[0],
              'after_unmatched_reference': len(ref_onsets)-counts[1],
              'metric': 'common_reference_voiced_onsets', 'resolution_ms': 1000/C.F0_FRAME_RATE}
    if len(ref_onsets):
        first = round(ref_onsets[0])
        result['first_reference_onset'] = {'time': first/C.F0_FRAME_RATE,
            'before_ms': round(rows[0][first][1], 1) if first in rows[0] else None,
            'after_ms': round(rows[1][first][1], 1) if first in rows[1] else None}
    if not result['available']:
        return {**result, 'reason': '共同可靠入点不足'}
    pairs = [{'time': round(key/C.F0_FRAME_RATE, 2),
              'before': round(rows[0][key][1], 1), 'after': round(rows[1][key][1], 1)} for key in shared]
    for label in ('before', 'after'):
        errors = np.abs([p[label] for p in pairs])
        result[label+'_median_ms'] = round(float(np.median(errors)), 1)
        result[label+'_p90_ms'] = round(float(np.percentile(errors, 90)), 1)
    result['pairs'] = pairs
    return result


def voice_features(signal, sr):
    """独立人声活动段 RMS 归一后的频谱与动态；纯推大音量不会改变频谱形状。"""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[None, :]
    hop = max(1, round(.1*sr))
    n = signal.shape[-1]
    if n < 4096 or audio.rms(signal) < 1e-8:
        return {'available': False}
    frames = np.pad(signal, ((0, 0), (0, (-n) % hop))).reshape(signal.shape[0], -1, hop)
    levels = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=(0, 2))+1e-16)
    active = levels > max(float(np.max(levels))*10**(-35/20), 1e-5)
    if not np.any(active):
        return {'available': False}
    reference = float(np.sqrt(np.mean(levels[active]**2)))
    window = np.hanning(4096)
    power = np.zeros(2049, dtype=np.float64)
    count = 0
    for start in range(0, n-4095, 65536):
        chunk = signal[:, start:min(n, start+65536+2048)]
        if chunk.shape[-1] < 4096:
            continue
        spectrum = np.lib.stride_tricks.sliding_window_view(chunk, 4096, axis=-1)[:, ::2048]
        transformed = np.fft.rfft(spectrum*window, axis=-1)
        power += np.mean(np.abs(transformed)**2, axis=0).sum(axis=0)
        count += transformed.shape[1]
    power /= max(count, 1)*np.sum(window)**2*reference**2
    hz = np.fft.rfftfreq(4096, 1/sr)
    centers = [f for f in CENTERS if f < sr/2]
    def energy(lo, hi):
        return float(10*np.log10(max(float(power[(hz >= lo) & (hz < hi)].sum()), 1e-14)))
    spectral = [energy(f/2**(1/6), f*2**(1/6)) for f in centers]
    db = 20*np.log10(np.maximum(levels/reference, 1e-8))
    indices = np.unique(np.linspace(0, len(db)-1, min(180, len(db))).astype(int))
    return {'available': True, 'centers': centers, 'spectrum': _numbers(spectral, 2),
        'bands': {name: round(energy(lo, min(hi, sr/2)), 2) for name, lo, hi in TONE_BANDS},
        'dynamic_range_db': round(float(np.percentile(db[active], 90)-np.percentile(db[active], 10)), 1),
        'envelope': _numbers(db[indices], 2), 'normalization': 'active_rms',
        'seconds': round(n/sr, 2)}


def stereo_scope(path, points=1000):
    """分块测量真实 L/R 能量，固定种子分层取样；声场对照去除纯增益差异。"""
    with sf.SoundFile(path) as stream:
        if not stream.frames:
            return {'available': False}
        count = min(points, stream.frames)
        indices = np.floor((np.arange(count)+np.random.default_rng(17).random(count))*stream.frames/count).astype(int)
        sampled = np.zeros((count, 2), dtype=np.float64)
        left2 = right2 = cross = 0.
        offset = 0
        while True:
            chunk = stream.read(65536, dtype='float64', always_2d=True)
            if not len(chunk):
                break
            left, right = chunk[:, 0], chunk[:, min(1, chunk.shape[1]-1)]
            left2 += float(left@left); right2 += float(right@right); cross += float(left@right)
            a, b = np.searchsorted(indices, [offset, offset+len(chunk)])
            local = indices[a:b]-offset
            sampled[a:b, 0], sampled[a:b, 1] = left[local], right[local]
            offset += len(chunk)
        energy = left2+right2
        if energy < 1e-12:
            return {'available': False}
        level = np.sqrt(energy/(2*offset))
        mid = (sampled[:, 0]+sampled[:, 1])/(2*level)
        side = (sampled[:, 0]-sampled[:, 1])/(2*level)
    return {'available': True, 'points': np.round(np.column_stack([side, mid]), 3).tolist(),
        'side_energy_percent': round(float(np.clip((energy-2*cross)/(2*energy), 0, 1)*100), 1),
        'correlation': round(cross/max(np.sqrt(left2*right2), 1e-12), 3)}


def results(report, capture):
    unavailable = {'available': False, 'reason': '此版本未保存结果分析，重新生成可查看'}
    if not capture or 'rendered_f0' not in capture:
        return {'pitch': unavailable, 'timing': unavailable, 'tone': unavailable}
    dry_end = len(capture['dry_f0']) + round(capture['move']*C.F0_FRAME_RATE)
    n = max(len(capture['ref_f0']), len(capture['rendered_f0']), dry_end)
    ref = _placed_frames(capture['ref_f0'], 0., n)
    before = _placed_frames(capture['dry_f0'], capture['move'], n)
    after = _placed_frames(capture['rendered_f0'], 0., n)
    original = _placed_frames(capture['input_f0'], report.get('comparison_global_move_seconds', capture['move']), n)
    source_tone = voice_features(capture['dry_input'], C.VOC_SR)
    final_tone = capture.get('report_vocal_features')
    if final_tone is None:
        final_tone = voice_features(capture['vocal'], C.VOC_SR)
    tone = {'available': source_tone['available'] and final_tone['available'],
            'before': source_tone, 'after': final_tone, 'normalization': 'active_rms',
            'position': 'vocal_before_spatial_returns'}
    if tone['available']:
        tone['changes'] = [{'name': name, 'lo': lo, 'hi': hi,
             'db': round(final_tone['bands'][name]-source_tone['bands'][name], 1)}
             for name, lo, hi in TONE_BANDS]
    pitch = pitch_result(before, after, ref)
    hop = round(C.VOC_SR/C.F0_FRAME_RATE)
    vocal = capture['vocal']
    padded = np.pad(vocal, (0, (-len(vocal)) % hop))
    energy = np.max(np.abs(padded.reshape(-1, hop)), axis=1)
    peak = _placed_frames(energy, 0., n)
    silent = (before > 0) & (peak < 1e-5)
    pitch['coverage']['output_near_silent_input_voiced_frames'] = int(np.sum(silent))
    pitch['coverage']['output_near_silent_intervals_seconds'] = _frame_intervals(silent)
    return {'pitch': pitch, 'timing': timing_result(original, after, ref), 'tone': tone}
