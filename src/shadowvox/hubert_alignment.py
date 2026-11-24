"""中文歌声对齐：本地唱词识别、音素定位、保序匹配与受限变时。"""
import json
from collections import defaultdict
from itertools import groupby

import numpy as np
import onnxruntime as ort
from scipy.optimize import LinearConstraint, minimize
from scipy.signal import resample_poly

from . import config as C
from .lyric_anchors import _is_han, _match
from .natural import runs
from .vendor.hubertfa.decoder import AlignmentDecoder
from .vendor.hubertfa.g2p import DictionaryG2P

WORD_WINDOW = .5
MIN_PAUSE = .18
PAUSE_RESERVE = .08
PHRASE_EDGE = .25
PHRASE_CONSENSUS = .08
MAX_PHRASE_MOVE = .45
MAX_LOCAL_SHIFT = .15
LOCAL_JITTER = .004
LARGE_RESIDUAL = .12


_PINYIN_FALLBACK = {'n': 'en', 'ng': 'en', 'hm': 'hen', 'hng': 'heng', 'm': 'mu', 'r': 'er'}


class ChineseAligner:
    def __init__(self):
        folder = C.MODELS/'hubertfa'
        self.vocab = json.loads((folder/'vocab.json').read_text(encoding='utf-8'))
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        self.sr = config['mel_spec_config']['sample_rate']
        self.hop = config['mel_spec_config']['hop_size']
        self.g2p = DictionaryG2P('zh' if self.vocab['language_prefix'] else None,
                                folder/self.vocab['dictionaries']['zh'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder/'model.onnx'), options,
                                            providers=['CPUExecutionProvider'])

    def decode(self, wave, sr, text, offset=0.):
        from pypinyin import lazy_pinyin
        text = ''.join(c for c in text if _is_han(c))
        raw_tokens = lazy_pinyin(text)
        tokens = []
        for token in raw_tokens:
            mapped = _PINYIN_FALLBACK.get(token, token)
            if mapped in self.g2p.dictionary:
                tokens.append(mapped)
        if len(tokens) < 2:
            return []
        phones, words, indices = self.g2p(' '.join(tokens))
        x = resample_poly(wave, self.sr, sr).astype(np.float32) if sr != self.sr else np.asarray(wave, np.float32)
        values = self.session.run(['ph_frame_logits', 'ph_edge_logits'], {'waveform': x[None]})
        decoded, confidence = AlignmentDecoder(self.vocab, self.sr, self.hop).decode(
            *values, len(x)/self.sr, phones, words, indices)
        if confidence < .35 or len(decoded) != len(tokens):
            return []
        return [dict(text=w.text, start=float(w.start)+offset, end=float(w.end)+offset,
                     final_start=float(w.phonemes[-1].start)+offset,
                     final_end=float(w.phonemes[-1].end)+offset) for w in decoded]


def _text(model, wave, sr, prompt=None):
    x = resample_poly(wave, 16000, sr).astype(np.float32)
    kwargs = dict(language='zh', temperature=0, condition_on_previous_text=False,
                  fp16=False, word_timestamps=True, verbose=None)
    if prompt:
        kwargs['initial_prompt'] = prompt
    result = model.transcribe(x, **kwargs)
    # 歌声的整窗 no_speech_prob 可偏高；已有可靠识别结果时不能据此删掉整句。
    return ''.join(s['text'] for s in result['segments']
                   if s.get('avg_logprob', 0) >= -1. and s.get('compression_ratio', 0) <= 2.4)


def _windows(dry_active, ref_active, duration):
    count = len(dry_active)
    ref_active = np.pad(ref_active[:count], (0, max(0, count-len(ref_active))))
    pauses = [(a+b)/20. for a,b in runs(~(dry_active | ref_active)) if b-a >= 2]
    start = 0.
    while duration-start > 18.:
        candidates = [t for t in pauses if start+8 <= t <= start+18]
        end = min(candidates, key=lambda t: abs(t-start-14)) if candidates else start+14
        yield start, end
        start = end
    yield start, duration


def word_groups(left, right, offset=0., scale=1.0):
    """只接受连续多个同音字；漏唱和多唱可跳过，不强制一一配齐。"""
    if not left or not right:
        return []
    vocab = {s:i for i,s in enumerate(sorted({w['text'] for w in left+right}))}
    a,b = (np.array([vocab[w['text']] for w in words], np.int32) for words in (left,right))
    ta,tb = (np.array([w['start'] for w in words]) for words in (left,right))
    ta_ref = scale * ta
    ia,ib = _match(a,b,a,b,ta_ref,tb,offset)
    groups, current = [], []
    for i,j in zip(ia,ib):
        if abs(tb[j]-ta_ref[i]-offset) > WORD_WINDOW:
            if current: groups.append(current)
            current = []
            continue
        if current and (i != current[-1][0]+1 or j != current[-1][1]+1 or ta[i]-ta[current[-1][0]] > 1.5):
            groups.append(current)
            current = []
        current.append((i,j))
    if current: groups.append(current)
    def supported(g):
        if len(g) < 2 or len({a[i] for i,_ in g}) < 2:
            return False
        if len(g) >= 3:
            return True
        (i,j),(k,l) = g
        pairs = [(left[i],right[j]),(left[k],right[l])]
        if any(not _usable_vowels(x,y) for x,y in pairs):
            return False
        if abs((right[j]['final_start']-scale*left[i]['final_start']) -
               (right[l]['final_start']-scale*left[k]['final_start'])) > PHRASE_CONSENSUS:
            return False

        def ambiguous(tokens, times, index, expected):
            for alternate in range(len(tokens)-1):
                if (alternate != index and tokens[alternate] == tokens[index]
                        and tokens[alternate+1] == tokens[index+1]
                        and np.max(abs(times[alternate:alternate+2]-expected)) <= WORD_WINDOW):
                    return True
            return False

        return not (ambiguous(a,ta,i,(tb[j:l+1]-offset)/scale) or ambiguous(b,tb,j,scale*ta[i:k+1]+offset))

    return [[(left[i],right[j]) for i,j in g] for g in groups if supported(g)]


def _usable_vowels(a,b):
    return min(a['final_end']-a['final_start'],b['final_end']-b['final_start']) >= .03


def anchors_for_words(words):
    anchors, trusted = [], []
    for a,b in words:
        if not _usable_vowels(a,b):
            continue
        if (a['final_start'] > a['start'] + .005
                and abs((b['start']-a['start'])-(b['final_start']-a['final_start'])) <= .10):
            anchors.append((a['start'],b['start']))
        anchors.append((a['final_start'],b['final_start']))
        trusted.append((a['final_start'],a['end']))
    return anchors, trusted


def fit(source, reference, strength=1., max_rate=.10, base_scale=1.0):
    """整段一次拟合；固定首尾，最多 150ms 位移，叠加全局速度比后总伸缩率不超上限。"""
    rate = float(np.clip(max_rate, 0, .5))
    amount = float(np.clip(strength, 0, 1))
    lo_r = max(0., base_scale - (1. - rate))
    hi_r = max(0., (1. + rate) - base_scale)
    if not rate or not amount or lo_r <= 1e-9 or hi_r <= 1e-9:
        return source.copy()
    edge_rate = min(lo_r, hi_r)
    ceiling = np.minimum(MAX_LOCAL_SHIFT, edge_rate*np.minimum(source-source[0], source[-1]-source))
    ds = np.diff(source)
    proposal = reference-source
    result = minimize(lambda d: float(np.sum((d-proposal)**2)), np.zeros(len(source)),
        jac=lambda d: 2*(d-proposal), method='SLSQP', bounds=list(zip(-ceiling,ceiling)),
        constraints=[LinearConstraint(np.diff(np.eye(len(source)),axis=0),-lo_r*ds,hi_r*ds)],
        options={'ftol':1e-11,'maxiter':200})
    if not result.success:
        raise RuntimeError('中文对齐时间拟合失败：'+result.message)
    return source+amount*result.x


def _phrase_moves(phrases, duration, amount, scale=1.0):
    moves = np.zeros(len(phrases))
    reasons = [p['status'] for p in phrases]
    for i,p in enumerate(phrases):
        if p['status'] != 'candidate':
            continue
        move = p['requested_move']*amount
        if abs(move) < 1e-7:
            reasons[i] = 'disabled'
            continue
        lo,hi = p['source_interval']
        before = scale*(lo-(phrases[i-1]['source_interval'][1] if i else 0.))
        after = scale*((phrases[i+1]['source_interval'][0] if i+1 < len(phrases) else duration)-hi)
        remaining = min(before+move,after-move)
        if min(before,after) < MIN_PAUSE-1e-7 or remaining < PAUSE_RESERVE-1e-7:
            reasons[i] = 'insufficient_pause'
            continue
        moves[i], reasons[i] = move, 'moved'
    conflicts = set()
    for i,(left,right) in enumerate(zip(phrases,phrases[1:])):
        if scale*(right['source_interval'][0]-left['source_interval'][1])+moves[i+1]-moves[i] < PAUSE_RESERVE-1e-7:
            conflicts.update((i,i+1))
    for i in conflicts:
        if moves[i]:
            moves[i], reasons[i] = 0., 'conflicting_moves'
    return moves, reasons


def plan_segments(plan, strength=1., max_rate=.10):
    amount = float(np.clip(strength,0.,1.))
    duration = plan['dry_seconds']
    eff_scale = float(plan.get('scale', 1.0)) if amount > 0. else 1.0
    phrases = plan.get('phrases') or [dict(source_interval=(0.,duration),status='unmatched')]
    moves,reasons = _phrase_moves(phrases,duration,amount,scale=eff_scale)
    phrase_timeline = bool(np.any(moves))
    render_phrases = phrases if phrase_timeline else [dict(source_interval=(0.,duration))]
    render_moves = moves if phrase_timeline else [0.]
    windows = defaultdict(list)
    selected = plan['windows'] if phrase_timeline else plan.get('continuous_windows',plan['windows'])
    for window in selected:
        windows[window.get('phrase',0) if phrase_timeline else 0].append(window)
    pieces, cursor, previous_move = [], 0., 0.

    def add(source,target,*,pause=False,move=0.,fit_failed=False):
        piece = dict(source=np.asarray(source),target=np.asarray(target),pause=pause,phrase_move=move,
                     protected=bool(np.max(abs(np.asarray(target)-source)) < 1e-7))
        if fit_failed:
            piece['render_fallback'] = 'local_fit_failed'
        pieces.append(piece)

    for i,(phrase,move) in enumerate(zip(render_phrases,render_moves)):
        lo,hi = phrase['source_interval']
        if lo > cursor:
            add([cursor,lo],[cursor+previous_move,lo+move],pause=True)
        cursor = lo
        for window in windows[i]:
            source,reference = window['source'],window['reference']
            if source[0] > cursor+1e-7:
                add([cursor,source[0]],[cursor+move,source[0]+move],move=move)
            # Remove the full phrase proposal before scaling its within-phrase correction.
            baseline = phrase['requested_move'] if move else 0.
            reference = reference-baseline
            # Large unresolved differences are not an instruction to distort a voiced phrase.
            if phrase_timeline:
                reference = np.where(abs(reference-source) <= MAX_LOCAL_SHIFT+1e-7,reference,source)
                if np.max(abs(reference[1:-1]-source[1:-1]),initial=0.) < LOCAL_JITTER:
                    reference = source
            failed = False
            try:
                target = fit(source,reference,amount,max_rate,base_scale=eff_scale)
            except RuntimeError:
                target,failed = source,True
            add(source,target+move,move=move,fit_failed=failed)
            cursor = source[-1]
        if hi > cursor:
            add([cursor,hi],[cursor+move,hi+move],move=move)
        cursor,previous_move = hi,move
    if cursor < duration:
        add([cursor,duration],[cursor+previous_move,duration],pause=True)
    # 同一条连续时标，句内连续变时；句间调整只借用低活动停顿。
    # 局部/句级变时按节奏强度缩放，但已确认的全局偏移与全局速度比应完整补偿（仅关闭节奏时保持原位）。
    move = plan['offset'] if amount > 0. else 0.
    placement = max(0., move)
    reference_shift = max(0., -move)
    for piece in pieces:
        piece['continuous'] = True
        piece['phrase_timeline'] = phrase_timeline
        local_delta = piece['target'] - piece['source']
        piece['target'] = piece['source'] * eff_scale + local_delta + placement
        piece['reference_shift'] = reference_shift
        piece['tempo_scale'] = eff_scale
    return pieces, summarize(pieces,plan,max_rate,moves,reasons)


def summarize(pieces, plan, max_rate, moves=(), reasons=()):
    voice_pieces = [p for p in pieces if not p.get('pause')]
    rates = np.concatenate([np.diff(p['target'])/np.diff(p['source']) for p in voice_pieces] or [np.ones(1)])
    rate = float(np.clip(max_rate, 0, .5))
    placement = float(pieces[0]['target'][0])
    reference_shift = pieces[0]['reference_shift']
    eff_scale = float(pieces[0].get('tempo_scale', 1.0))
    move = placement-reference_shift
    median_move = move + (eff_scale - 1.0) * (plan['dry_seconds'] / 2.0)
    shift = max(float(np.max(abs(p['target']-p['source']*eff_scale-placement))) for p in pieces)
    protected = [[float(p['target'][0]),float(p['target'][-1])] for p in pieces if p['protected']]
    local_shift = max((float(np.max(abs(p['target']-p['source']*eff_scale-placement-p.get('phrase_move',0.))))
                       for p in voice_pieces),default=0.)
    phrase_timeline = pieces[0].get('phrase_timeline')
    stats = dict(algorithm='hubertfa-zh+signalsmith-v1', alignment_basis='chinese-phonemes',
        language='zh', rendering='pause_bounded_phrases' if phrase_timeline else 'continuous',
        protected_waveform_exact=False,
        lyric_anchors=sum(len(w['source'])-2 for w in (plan['windows'] if phrase_timeline else plan.get('continuous_windows',plan['windows']))),
        runs=len(pieces), reference_shift_seconds=reference_shift, placement_offset_seconds=placement,
        median_offset_seconds=median_move, global_offset_seconds=move,
        global_tempo_scale=round(eff_scale, 6),
        global_offset_reliable=bool(plan['offset'] or abs(plan.get('scale', 1.0) - 1.0) > 1e-5),
        reference_gaps=0, source_gaps=0,
        extra_duration_seconds=max(0.,plan['dry_seconds']*eff_scale+move-plan['ref_seconds']),
        protected_intervals_seconds=protected, protected_seconds=round(sum(b-a for a,b in protected),3),
        active_intervals_seconds=[[float(p['target'][0]),float(p['target'][-1])] for p in voice_pieces if not p['protected']],
        max_rate=rate, max_local_shift_seconds=MAX_LOCAL_SHIFT,
        rate_min=round(float(rates.min()),6), rate_max=round(float(rates.max()),6),
        rate_scope='vocal_phrases' if phrase_timeline else 'timeline',
        max_applied_shift_ms=round(shift*1000,3), max_intra_phrase_shift_ms=round(local_shift*1000,3),
        rate_clipped_ratio=round(float(np.mean(abs(rates-1) >= rate-1e-7)),3) if rate else 0.,
        applied=shift > 1e-4 or abs(move) > 1e-4 or abs(eff_scale - 1.0) > 1e-5,
        render_fallbacks=sum('render_fallback' in p for p in pieces),
        order_preserved=True, continuous_source_coverage=True)
    source = np.concatenate([p['source'][:-1] for p in pieces]+[pieces[-1]['source'][-1:]])
    output = np.concatenate([p['target'][:-1] for p in pieces]+[pieces[-1]['target'][-1:]])
    mapped = lambda t: np.interp(t,source,output)
    first_phrase, phrase_moves, unmatched, preserve = None, [], [], []
    for phrase,phrase_move,status in zip(plan.get('phrases',()),moves,reasons):
        lo,hi = phrase['source_interval']
        interval = mapped([lo,hi]).tolist()
        item = dict(target_interval_seconds=interval, phrase_move_ms=round(float(phrase_move)*1000,3), status=status,
                    source_onset_seconds=None, reference_onset_seconds=None,
                    residual_before_ms=None, residual_after_ms=None,onset_basis=phrase.get('onset_basis'))
        if phrase['first'] is not None:
            a,b = phrase['first']
            item.update(source_onset_seconds=a,reference_onset_seconds=b,
                        residual_before_ms=round((a-b)*1000,3),
                        residual_after_ms=round(float(mapped(a)-b-reference_shift)*1000,3))
        if first_phrase is None:
            first_phrase = item
        if phrase_move:
            phrase_moves.append(item)
        words = phrase['words']
        if len(words) < 2:
            unmatched.append(interval)
        else:
            # Only the confidently matched words with residual can override a reference gap.
            for a,b in words:
                match = mapped([a['start'],a['end']])
                if np.max(abs(match-[b['start']+reference_shift,b['end']+reference_shift])) >= LARGE_RESIDUAL:
                    preserve.append(match.tolist())
    stats.update(first_phrase=first_phrase,phrase_moves=phrase_moves,
                 unmatched_intervals_seconds=unmatched,vocal_preserve_intervals_seconds=preserve)
    return stats


def _guided_reference(reference, sr, aligner, ref_voiced, lyrics, log):
    """Use each line to bound forced alignment of the actual reference audio."""
    words = []
    duration = len(reference)/sr
    for index, (start, text) in enumerate(lyrics):
        if start >= duration:
            break
        end = lyrics[index+1][0] if index+1 < len(lyrics) else duration
        left = max(0., start-.6, words[-1]['end'] if words else 0.)
        right = min(duration, end+.2, start+12.)
        if right-left < .35 or not ref_voiced[int(left*10):int(np.ceil(right*10))].any():
            continue
        log('[节奏] 用 LRC 定位原唱字句')
        line = aligner.decode(reference[round(left*sr):round(right*sr)], sr, text, left)
        if line and abs(line[0]['start']-start) <= 1.:
            words.extend(line)
    return words


def _matched_groups(dry, reference, sr, model, aligner, voiced, ref_voiced, offset, log,
                    guided_words, guided_windows, lyrics=(), scale=1.0):
    duration, reference_duration = len(dry)/sr, len(reference)/sr
    windows = list(_windows(voiced,ref_voiced,duration))
    for number,(start,end) in enumerate(windows,1):
        log(f'[节奏] 中文字句定位 {number}/{len(windows)}')
        a,b = max(0.,start-.5), min(duration,end+.5)
        c,d = max(0.,scale*start+offset-.5), min(reference_duration,scale*end+offset+.5)
        if min(b-a,d-c) < .5 or not voiced[int(start*10):int(np.ceil(end*10))].any():
            continue
        left, right = dry[round(a*sr):round(b*sr)], reference[round(c*sr):round(d*sr)]
        dw = aligner.decode(left,sr,_text(model,left,sr),a)

        def matches(dry_words, ref_words):
            for group in word_groups(dry_words, ref_words, offset, scale=scale):
                group = [(l,r) for l,r in group if start <= l['start'] < end
                         and a+.03 < l['start'] and l['end'] < b-.03
                         and c+.03 < r['start'] and r['end'] < d-.03]
                if len(group) >= 2:
                    yield group

        if guided_words:
            cues = [w for w in guided_words if c+.03 < w['start'] and w['end'] < d-.03]
            groups = list(matches(dw, cues))
            words_in_window = sum(start <= w['start'] < end for w in dw)
            matched_words = sum(_usable_vowels(l, r) for group in groups for l, r in group)
            if not (matched_words >= 2 and matched_words*2 >= words_in_window) and lyrics and cues:
                prompt = ''.join(text for ts, text in lyrics if c - 2. <= ts <= d + 1.)[:120]
                if prompt:
                    prompted_dw = aligner.decode(left, sr, _text(model, left, sr, prompt=prompt), a)
                    p_groups = list(matches(prompted_dw, cues))
                    p_words = sum(start <= w['start'] < end for w in prompted_dw)
                    p_matched = sum(_usable_vowels(l, r) for group in p_groups for l, r in group)
                    if p_matched >= 3 and p_matched * 2 >= p_words:
                        groups, matched_words, words_in_window = p_groups, p_matched, p_words
            if matched_words >= 2 and matched_words*2 >= words_in_window:
                guided_windows.add(number)
                for group in groups:
                    yield start,end,group
                continue
        rw = aligner.decode(right,sr,_text(model,right,sr),c)
        for group in matches(dw, rw):
            yield start,end,group


def _phrase_spans(active,duration):
    spans = []
    for a,b in runs(active):
        lo,hi = a/10.,min(b/10.,duration)
        if lo >= hi:
            continue
        if spans and lo-spans[-1][1] < MIN_PAUSE-1e-7:
            spans[-1] = spans[-1][0],hi
        else:
            spans.append((lo,hi))
    return spans


def _span_locator(spans):
    starts = [a for a,_ in spans]

    def locate(time):
        i = int(np.searchsorted(starts,time,side='right'))-1
        return i if i >= 0 and time < spans[i][1] else None

    return locate


def _confirmed_offset(groups, hint, spans=None, min_offset=.12):
    if not hint:
        return 0.
    estimates = [(group[0][0]['start'], np.median([b['final_start']-a['final_start'] for a,b in group]))
                 for _,_,group in groups]
    close = [(t,d) for t,d in estimates if abs(d-hint) <= .22]
    if len(close) < 2:
        return 0.
    median = float(np.median([d for _,d in close]))
    consistent = [t for t,d in close if abs(d-median) <= .12]
    if abs(median) < min_offset or len(consistent) < 2 or consistent[-1]-consistent[0] < 2.:
        return 0.
    if spans is not None:
        locate = _span_locator(spans)
        if len({locate(t+.05) for t in consistent}-{None}) < 2:
            return 0.
    return median


def _confirmed_affine(groups, acoustic, hint_offset, spans=None):
    """结合中文歌词匹配与多窗声学共识确认全局 (scale, offset, confirmed)。"""
    if acoustic and acoustic.get('scale_applied'):
        ac_scale = float(acoustic['scale'])
        ac_offset = float(acoustic['offset'])
        word_pts = [
            (a['final_start'], b['final_start'])
            for _, _, group in groups for a, b in group
            if abs(b['final_start'] - (ac_scale * a['final_start'] + ac_offset)) <= .25
        ]
        if len(word_pts) >= 6:
            xs = np.array([p[0] for p in word_pts], dtype=np.float64)
            ys = np.array([p[1] for p in word_pts], dtype=np.float64)
            resid = ys - (ac_scale * xs + ac_offset)
            inliers = np.abs(resid) <= .10
            if int(inliers.sum()) >= 6 and float(np.ptp(xs[inliers])) >= 40.:
                w_scale, w_offset = [float(v) for v in np.polyfit(xs[inliers], ys[inliers], 1)]
                w_res = float(np.median(np.abs(ys[inliers] - (w_scale * xs[inliers] + w_offset))))
                if abs(w_scale - ac_scale) <= .004 and w_res <= .05:
                    return w_scale, w_offset, True
            if float(np.median(np.abs(resid))) <= .10:
                return ac_scale, ac_offset, True
        # 歌词稀疏（如粤语/弱人声）但声学多窗线性漂移高度一致时直接采纳声学共识。
        if (acoustic.get('inliers', 0) >= 6
                and acoustic['inliers'] / max(1, acoustic.get('total_windows', 1)) >= .80
                and acoustic.get('span_seconds', 0.) >= 50.
                and acoustic.get('linear_residual_ms', 999.) <= 20.):
            return ac_scale, ac_offset, True

    ac_reliable = bool(acoustic and acoustic.get('reliable') and not acoustic.get('scale_applied'))
    min_off = .06 if ac_reliable else .12
    word_offset = _confirmed_offset(groups, hint_offset, spans, min_offset=min_off)
    if word_offset:
        return 1.0, word_offset, True
    if (ac_reliable
            and acoustic.get('inliers', 0) >= 4
            and acoustic['inliers'] / max(1, acoustic.get('total_windows', 1)) >= .75
            and acoustic.get('span_seconds', 0.) >= 40.
            and acoustic.get('const_residual_ms', 999.) <= 30.
            and abs(float(acoustic.get('offset', 0.))) >= .06):
        return 1.0, float(acoustic['offset']), True
    return 1.0, 0.0, False


def _phrase_evidence(span,words,ref_spans,offset,scale=1.0):
    phrase = dict(source_interval=span,words=words,first=None,requested_move=0.,status='unmatched')
    if len(words) < 2:
        return phrase
    locate_ref = _span_locator(ref_spans)
    ref_indices = {locate_ref((b['final_start']+b['final_end'])/2) for _,b in words}
    a0,b0 = words[0]
    a1,b1 = words[-1]
    first_ref = locate_ref((b0['final_start']+b0['final_end'])/2)
    phrase['status'] = 'uncertain_boundaries'
    if first_ref is None or len(ref_indices) != 1:
        return phrase
    ref_lo,ref_hi = ref_spans[first_ref]
    if abs(a0['start']-span[0]) <= PHRASE_EDGE and abs(b0['start']-ref_lo) <= PHRASE_EDGE:
        phrase.update(first=(float(a0['start']),float(b0['start'])),onset_basis='matched_lyrics')
    onset = np.array([b['start']-(scale*a['start']+offset) for a,b in words])
    vowel = np.array([b['final_start']-(scale*a['final_start']+offset) for a,b in words])
    stable = np.where(abs(onset-vowel) <= .10,onset,vowel)
    center = float(np.median(stable))
    inliers = abs(stable-center) <= PHRASE_CONSENSUS
    if inliers.sum() < max(2,np.ceil(.7*len(words))) or not (inliers[0] and inliers[-1]):
        phrase['status'] = 'inconsistent_words'
        return phrase
    delta = float(np.median(stable[inliers]))
    # If recognition missed an edge, the two activity boundaries must support the same displacement.
    corroborated = (len({a['text'] for a,_ in words}) >= 3 and a1['end']-a0['start'] >= .5*(span[1]-span[0]) and
                    abs(ref_lo-(scale*span[0]+offset)-delta) <= .12 and abs(ref_hi-(scale*span[1]+offset)-delta) <= .12)
    if not corroborated and (phrase['first'] is None or abs(a1['end']-span[1]) > PHRASE_EDGE or abs(b1['end']-ref_hi) > PHRASE_EDGE):
        return phrase
    if phrase['first'] is None:
        phrase.update(first=(float(span[0]),float(ref_lo)),onset_basis='activity_supported_by_lyrics')
    phrase['requested_move'] = delta
    phrase['status'] = ('within_tolerance' if abs(delta) < .06 else
                        'outside_move_limit' if abs(delta) > MAX_PHRASE_MOVE else 'candidate')
    return phrase


def _acoustic_onset_groups(dry_f0, ref_f0, offset, existing_groups, duration, ref_duration,
                           scale=1.0, key_semitones=0):
    """在已确认全局偏移/速度后，用经旋律校验的浊音起音点补充 Whisper 漏识的句内入点锚点。"""
    if dry_f0 is None or ref_f0 is None or len(dry_f0) < 30 or len(ref_f0) < 30:
        return []
    from .f0 import cent
    from .rhythm import FPS, onset_alignment

    # onset_alignment 的 global_offset 语义为「干声比原唱晚多少秒」，与此处 offset 互为相反数。
    ta, dv = onset_alignment(dry_f0, ref_f0, -offset, skip=0.14, scale=scale)
    if len(ta) < 2:
        return []
    tb = scale * ta + dv + offset  # 原唱时间轴上的真实起音秒数
    dry_c, ref_c = cent(dry_f0), cent(ref_f0)
    key_cents = float(key_semitones) * 100.0

    # 收集已有歌词锚点区间，避免与高可信音素锚点冲突或交叉。
    covered = []
    for _, _, group in existing_groups:
        if group:
            covered.append((group[0][0]['start'] - 0.12, group[-1][0]['end'] + 0.12))

    def is_covered(t):
        return any(lo <= t <= hi for lo, hi in covered)

    # 逐点做旋律一致性核验：起音后 150ms 内双轨有声且八度折叠（扣除已确认异调）后同旋律。
    verified = []
    for td, tr in zip(ta, tb):
        if abs(tr - (scale * td + offset)) > 0.18:
            continue
        if td < 0.05 or td > duration - 0.15 or tr < 0.05 or tr > ref_duration - 0.15:
            continue
        if is_covered(td):
            continue
        i0 = int(round(td * FPS))
        j0 = int(round(tr * FPS))
        span = min(15, len(dry_f0) - i0, len(ref_f0) - j0)
        if span < 6:
            continue
        d_seg, r_seg = dry_f0[i0:i0 + span], ref_f0[j0:j0 + span]
        both = (d_seg > 0) & (r_seg > 0)
        if int(both.sum()) < 5:
            continue
        diff = dry_c[i0:i0 + span][both] - ref_c[j0:j0 + span][both] - key_cents
        oct_shift = float(np.rint(np.median(diff) / 1200.0))
        err = float(np.median(np.abs(diff - 1200.0 * oct_shift)))
        if err > 240.0:
            continue
        end_d = min(duration - 0.02, td + max(0.08, float(both.sum()) / FPS))
        end_r = min(ref_duration - 0.02, tr + max(0.08, float(both.sum()) / FPS))
        verified.append((float(td), float(tr), end_d, end_r))

    if len(verified) < 2:
        return []

    # 按停顿（>1.6s）和局部偏移一致性聚类成独立声学入点组（至少 2 个相邻一致入点）。
    groups, current = [], []
    for item in verified:
        td, tr, _, _ = item
        if current:
            prev_td, prev_tr, _, _ = current[-1]
            if td - prev_td > 1.6 or tr <= prev_tr + 0.04 or abs((tr - scale * td) - (prev_tr - scale * prev_td)) > 0.11:
                if len(current) >= 2:
                    groups.append(current)
                current = []
        current.append(item)
    if len(current) >= 2:
        groups.append(current)

    extra = []
    for g in groups:
        pairs = []
        for k, (td, tr, ed, er) in enumerate(g):
            w_dry = dict(text=f'_ao{k}', start=td, end=ed, final_start=td, final_end=ed, acoustic_onset=True)
            w_ref = dict(text=f'_ao{k}', start=tr, end=er, final_start=tr, final_end=er, acoustic_onset=True)
            pairs.append((w_dry, w_ref))
        win_start = max(0.0, g[0][0] - 0.6)
        win_end = min(duration, g[-1][2] + 0.6)
        extra.append((win_start, win_end, pairs))
    return extra


def _add_window(candidates,words,start,end,offset,phrase=None,scale=1.0):
    if len(words) < 2:
        return
    pairs, trusted = anchors_for_words(words)
    minimum = 2 if len(words) == 2 or phrase is not None else 3
    if len(pairs) < minimum:
        return
    # 存储为干声坐标下的局部残差目标：x + (y - (scale * x + offset))；scale=1 时即 y - offset。
    pairs = [(x, x + (y - (scale * x + offset))) for x, y in pairs]
    # 留足 0.60s 窗边缓冲，避免首尾字被 max_rate*(source-source[0]) 卡死在 35ms 内。
    lo = max(start,words[0][0]['start']-.60)
    hi = min(end,words[-1][0]['end']+.60)
    merge = candidates and candidates[-1].get('phrase') == phrase and lo < candidates[-1]['end']
    if merge:
        previous = candidates[-1]
        lo = previous['source'][0]
        pairs = list(zip(previous['source'][1:-1],previous['reference'][1:-1]))+pairs
        trusted = previous['trusted']+trusted
    src, dst = [lo], [lo]
    for x,y in pairs:
        ref_inside = y > dst[-1]+.005 and y < hi-.005
        if phrase is not None:
            ref_inside = len(dst) == 1 or y > dst[-1]+.005
        if x > src[-1]+.005 and x < hi-.005 and ref_inside:
            src.append(x); dst.append(y)
    if len(src) <= minimum:
        return
    if merge:
        candidates.pop()
    candidates.append(dict(source=np.r_[src,hi],reference=np.r_[dst,hi],end=hi,trusted=trusted,phrase=phrase))


def build_plan(groups, voiced, ref_voiced, duration, reference_duration, offset=0., scale=1.0):
    spans = _phrase_spans(voiced,duration)
    ref_spans = _phrase_spans(ref_voiced,reference_duration)
    locate = _span_locator(spans)
    candidates, continuous, pitch_pairs, all_words = [], [], [], defaultdict(list)

    def phrase_index(pair):
        a,_ = pair
        return locate((a['final_start']+a['final_end'])/2)

    for start,end,group in sorted(groups, key=lambda item: item[2][0][0]['start'] if item[2] else item[0]):
        _add_window(continuous,group,start,end,offset,scale=scale)
        usable = [(a,b) for a,b in group if _usable_vowels(a,b)]
        for idx_u, (a, b) in enumerate(usable):
            if a.get('acoustic_onset'):
                continue
            # 若连续字音的浊声母短且时序一致，纳入音高对应以避免连音字间回弹归零。
            src_start, ref_start = a['final_start'], b['final_start']
            if idx_u > 0:
                prev_a, prev_b = usable[idx_u - 1]
                gap_a = a['final_start'] - prev_a['final_end']
                gap_b = b['final_start'] - prev_b['final_end']
                if (0. < gap_a <= .08 and 0. < gap_b <= .08
                        and abs((b['start'] - a['start']) - (b['final_start'] - a['final_start'])) <= .06):
                    src_start = max(prev_a['final_end'], a['start'])
                    ref_start = max(prev_b['final_end'], b['start'])
            pitch_pairs.append((src_start, a['final_end'], ref_start, b['final_end']))
        for phrase, chunk in groupby(usable,key=phrase_index):
            words = list(chunk)
            if phrase is None:
                continue
            for a,b in words:
                previous = all_words[phrase]
                if not previous or a['final_start'] > previous[-1][0]['final_start']+.02:
                    previous.append((a,b))
            _add_window(candidates,words,max(start,spans[phrase][0]),min(end,spans[phrase][1]),offset,phrase,scale=scale)
    evidence = np.zeros(len(voiced),bool)
    grid = np.arange(len(voiced))*.1
    source, reference_times = [0.], [0.]
    for window in continuous:
        for a,b in window['trusted']:
            evidence |= (grid >= max(a,window['source'][0])) & (grid <= min(b,window['source'][-1]))
        for a,b in zip(window['source'],window['reference']):
            if a > source[-1]+1e-7:
                source.append(a); reference_times.append(b)
    if source[-1] < duration: source.append(duration);reference_times.append(duration)
    phrases = [_phrase_evidence(span,all_words[i],ref_spans,offset,scale=scale) for i,span in enumerate(spans)]
    return dict(backend='hubertfa_zh',windows=candidates,continuous_windows=continuous,phrases=phrases,
        dry_seconds=duration,ref_seconds=reference_duration,
        dry_t=np.array(source),ref_t=np.array(reference_times),voiced=voiced,offset=offset,scale=scale,anchors=[],
        pitch_pairs=np.asarray(sorted(pitch_pairs)).reshape(-1,4),
        acoustic_trusted=evidence & voiced,correspondence_trusted=evidence & voiced)


def align(dry, reference, sr, log=print, strength=1., capture=None, max_rate=.10, offset_hint=0., *, lyrics=(),
          dry_f0=None, ref_f0=None):
    import torch
    import whisper
    from .full_align import phrase_activity, render_segments
    from .rhythm import estimate_affine_alignment
    torch.set_num_threads(min(8,torch.get_num_threads()))
    log('[节奏] 加载内置中文歌词识别和歌声音素模型')
    model = whisper.load_model(str(C.MODELS/'whisper-base.pt'),device='cpu')
    aligner = ChineseAligner()
    voiced, ref_voiced = phrase_activity(dry,sr), phrase_activity(reference,sr)
    duration, reference_duration = len(dry)/sr, len(reference)/sr
    guided_words = _guided_reference(reference,sr,aligner,ref_voiced,lyrics,log) if lyrics else ()
    guided_windows = set()
    acoustic = estimate_affine_alignment(dry_f0, ref_f0) if (dry_f0 is not None and ref_f0 is not None) else {}
    if acoustic.get('scale_applied'):
        hint_scale = float(acoustic['scale'])
        hint_offset = float(acoustic['offset'])
        log(f"[节奏] 检测到线性速度漂移：速度比 {hint_scale:.5f}（{(hint_scale-1)*100:+.2f}%），"
            f"首端偏移 {hint_offset*1000:+.0f}ms（残差 {acoustic['const_residual_ms']:.0f} → {acoustic['linear_residual_ms']:.1f}ms）")
    elif acoustic.get('reliable') and abs(float(acoustic.get('offset', 0.))) >= .06:
        hint_scale = 1.0
        hint_offset = float(acoustic['offset'])
    else:
        hint_scale = 1.0
        hint_offset = offset_hint if abs(offset_hint) >= .18 else 0.
    groups = list(_matched_groups(dry,reference,sr,model,aligner,voiced,ref_voiced,hint_offset,log,
                                  guided_words,guided_windows,lyrics=lyrics,scale=hint_scale))
    spans = _phrase_spans(voiced,duration)
    scale, offset, confirmed = _confirmed_affine(groups, acoustic, hint_offset, spans)
    if (hint_offset or abs(hint_scale - 1.0) > 1e-6) and not confirmed:
        guided_windows.clear()
        scale, offset = 1.0, 0.0
        groups = list(_matched_groups(dry,reference,sr,model,aligner,voiced,ref_voiced,0.,log,
                                      guided_words,guided_windows,lyrics=lyrics,scale=1.0))
    # 补充经过旋律验证的浊音起音点，覆盖歌词识别漏字的句内入点。
    key_semi = int(acoustic.get('key_semitones', 0)) if acoustic.get('reliable') else 0
    acoustic_groups = _acoustic_onset_groups(
        dry_f0, ref_f0, offset, groups, duration, reference_duration, scale=scale, key_semitones=key_semi)
    if acoustic_groups:
        groups = groups + acoustic_groups
    plan = build_plan(groups,voiced,ref_voiced,duration,reference_duration,offset,scale=scale)
    pieces,stats = plan_segments(plan,strength,max_rate)
    end = max(round(pieces[-1]['target'][-1]*sr),len(reference)+round(stats['reference_shift_seconds']*sr))
    out,_ = render_segments(dry,pieces,sr,end=end)
    if any('render_fallback' in p for p in pieces):
        moves,reasons = _phrase_moves(plan['phrases'],duration,float(np.clip(strength,0.,1.)),scale=scale if strength > 0. else 1.0)
        stats = summarize(pieces,plan,max_rate,moves,reasons)
    if lyrics:
        stats['lrc_windows'] = len(guided_windows)
    stats['acoustic_onset_anchors'] = sum(len(g) for _, _, g in acoustic_groups)
    if acoustic.get('reliable'):
        stats['acoustic_global'] = acoustic
    log(f"[节奏] 中文音素+声学入点锚点 {stats['lyric_anchors']}（声学补充 {stats['acoustic_onset_anchors']}），实际变时 {len(stats['active_intervals_seconds'])} 段")
    if capture is not None: capture.update(rhythm_plan=plan,rhythm_pieces=pieces)
    return out,stats
