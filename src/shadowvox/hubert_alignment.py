"""中文歌声对齐：本地唱词识别、音素定位、保序匹配与受限变时。"""
import json

import numpy as np
import onnxruntime as ort
from scipy.optimize import LinearConstraint, minimize
from scipy.signal import resample_poly

from . import config as C
from .lyric_anchors import _is_han, _match
from .natural import runs
from .vendor.hubertfa.decoder import AlignmentDecoder
from .vendor.hubertfa.g2p import DictionaryG2P


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
        tokens = lazy_pinyin(text)
        if len(tokens) < 3 or any(t not in self.g2p.dictionary for t in tokens):
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


def _text(model, wave, sr):
    x = resample_poly(wave, 16000, sr).astype(np.float32)
    result = model.transcribe(x, language='zh', temperature=0, condition_on_previous_text=False,
                             fp16=False, word_timestamps=True, verbose=None)
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


def word_groups(left, right):
    """只接受连续多个同音字；漏唱和多唱可跳过，不强制一一配齐。"""
    if not left or not right:
        return []
    vocab = {s:i for i,s in enumerate(sorted({w['text'] for w in left+right}))}
    a,b = (np.array([vocab[w['text']] for w in words], np.int32) for words in (left,right))
    ta,tb = (np.array([w['start'] for w in words]) for words in (left,right))
    ia,ib = _match(a,b,a,b,ta,tb,0.)
    groups, current = [], []
    for i,j in zip(ia,ib):
        if abs(ta[i]-tb[j]) > .5:
            if current: groups.append(current)
            current = []
            continue
        if current and (i != current[-1][0]+1 or j != current[-1][1]+1 or ta[i]-ta[current[-1][0]] > 1.5):
            groups.append(current)
            current = []
        current.append((i,j))
    if current: groups.append(current)
    return [[(left[i],right[j]) for i,j in g] for g in groups
            if len(g) >= 3 and len({left[i]['text'] for i,j in g}) >= 2]


def anchors_for_words(words):
    anchors, trusted = [], []
    for a,b in words:
        if min(a['final_end']-a['final_start'], b['final_end']-b['final_start']) < .03:
            continue
        if abs((b['start']-a['start'])-(b['final_start']-a['final_start'])) <= .10:
            anchors.append((a['start'],b['start']))
        anchors.append((a['final_start'],b['final_start']))
        trusted.append((a['final_start'],a['end']))
    return anchors, trusted


def fit(source, reference, strength=1., max_rate=.10):
    """整段一次拟合；固定首尾，最多 150ms 位移，不超过给定伸缩率。"""
    rate = float(np.clip(max_rate, 0, .5))
    amount = float(np.clip(strength, 0, 1))
    if not rate or not amount:
        return source.copy()
    ceiling = np.minimum(.15, rate*np.minimum(source-source[0], source[-1]-source))
    limits = rate*np.diff(source)
    proposal = reference-source
    result = minimize(lambda d: float(np.sum((d-proposal)**2)), np.zeros(len(source)),
        jac=lambda d: 2*(d-proposal), method='SLSQP', bounds=list(zip(-ceiling,ceiling)),
        constraints=[LinearConstraint(np.diff(np.eye(len(source)),axis=0),-limits,limits)],
        options={'ftol':1e-11,'maxiter':200})
    if not result.success:
        raise RuntimeError('中文对齐时间拟合失败：'+result.message)
    return source+amount*result.x


def plan_segments(plan, strength=1., max_rate=.10):
    pieces, cursor = [], 0.
    for window in plan['windows']:
        source, reference = window['source'], window['reference']
        if source[0] > cursor:
            gap = np.array([cursor,source[0]])
            pieces.append(dict(source=gap,target=gap.copy(),protected=True))
        target = fit(source,reference,strength,max_rate)
        pieces.append(dict(source=source.copy(),target=target,protected=bool(np.max(abs(target-source)) < 1e-7)))
        cursor = source[-1]
    if cursor < plan['dry_seconds']:
        gap = np.array([cursor,plan['dry_seconds']])
        pieces.append(dict(source=gap,target=gap.copy(),protected=True))
    # 同一条连续时标一次变时，避免把句内拼接回退又变成旧版的稀疏对齐。
    for piece in pieces: piece['continuous'] = True
    return pieces, summarize(pieces,plan,max_rate)


def summarize(pieces, plan, max_rate):
    rates = np.concatenate([np.diff(p['target'])/np.diff(p['source']) for p in pieces])
    rate = float(np.clip(max_rate, 0, .5))
    shift = max(float(np.max(abs(p['target']-p['source']))) for p in pieces)
    protected = [[float(p['target'][0]),float(p['target'][-1])] for p in pieces if p['protected']]
    return dict(algorithm='hubertfa-zh+rubberband-r3-v1', alignment_basis='chinese-phonemes',
        language='zh', rendering='continuous', protected_waveform_exact=False,
        lyric_anchors=sum(len(w['source'])-2 for w in plan['windows']),
        runs=len(pieces), reference_shift_seconds=0., placement_offset_seconds=0., median_offset_seconds=0.,
        global_offset_reliable=False, reference_gaps=0, source_gaps=0,
        extra_duration_seconds=max(0.,plan['dry_seconds']-plan['ref_seconds']),
        protected_intervals_seconds=protected, protected_seconds=round(sum(b-a for a,b in protected),3),
        active_intervals_seconds=[[float(p['target'][0]),float(p['target'][-1])] for p in pieces if not p['protected']],
        max_rate=rate, max_local_shift_seconds=.15,
        rate_min=round(float(rates.min()),6), rate_max=round(float(rates.max()),6),
        max_applied_shift_ms=round(shift*1000,3),
        rate_clipped_ratio=round(float(np.mean(abs(rates-1) >= rate-1e-7)),3) if rate else 0.,
        applied=shift > 1e-4,
        render_fallbacks=sum('render_fallback' in p for p in pieces),
        order_preserved=True, continuous_source_coverage=True)


def align(dry, reference, sr, log=print, strength=1., capture=None, max_rate=.10):
    import torch
    import whisper
    from .full_align import singing, render_segments
    torch.set_num_threads(min(8,torch.get_num_threads()))
    log('[节奏] 加载内置中文歌词识别和歌声音素模型')
    model = whisper.load_model(str(C.MODELS/'whisper-base.pt'),device='cpu')
    aligner = ChineseAligner()
    voiced, ref_voiced = singing(dry,sr), singing(reference,sr)
    duration, reference_duration = len(dry)/sr, len(reference)/sr
    windows = list(_windows(voiced,ref_voiced,duration))
    candidates = []
    for number,(start,end) in enumerate(windows,1):
        log(f'[节奏] 中文字句定位 {number}/{len(windows)}')
        a,b = max(0.,start-.5), min(reference_duration,duration,end+.5)
        if b-a < .5 or not voiced[int(start*10):int(np.ceil(end*10))].any():
            continue
        left, right = dry[round(a*sr):round(b*sr)], reference[round(a*sr):round(b*sr)]
        dw = aligner.decode(left,sr,_text(model,left,sr),a)
        rw = aligner.decode(right,sr,_text(model,right,sr),a)
        for group in word_groups(dw,rw):
            # 重叠窗只认领本窗核心内的字，不将边缘截断字作可靠锚点。
            group = [(l,r) for l,r in group if start <= l['start'] < end
                     and min(l['start'],r['start']) > a+.03 and max(l['end'],r['end']) < b-.03]
            if len(group) < 3: continue
            pairs, trusted = anchors_for_words(group)
            if len(pairs) < 3: continue
            lo, hi = max(start,group[0][0]['start']-.35), min(end,group[-1][0]['end']+.35)
            if candidates and lo < candidates[-1]['end']:
                previous = candidates.pop()
                lo = previous['source'][0]
                pairs = list(zip(previous['source'][1:-1],previous['reference'][1:-1]))+pairs
                trusted = previous['trusted']+trusted
            src, dst = [lo], [lo]
            for x,y in pairs:
                if x > src[-1]+.005 and y > dst[-1]+.005 and max(x,y) < hi-.005:
                    src.append(x); dst.append(y)
            if len(src) < 4 or hi <= src[-1]+.005: continue
            candidates.append(dict(source=np.r_[src,hi],reference=np.r_[dst,hi],end=hi,trusted=trusted))
    evidence = np.zeros(len(voiced),bool)
    grid = np.arange(len(voiced))*.1
    source, reference_times = [0.], [0.]
    for window in candidates:
        for a,b in window['trusted']:
            evidence |= (grid >= max(a,window['source'][0])) & (grid <= min(b,window['source'][-1]))
        for a,b in zip(window['source'],window['reference']):
            if a > source[-1]+1e-7:
                source.append(a); reference_times.append(b)
    if source[-1] < duration: source.append(duration);reference_times.append(duration)
    plan = dict(backend='hubertfa_zh',windows=candidates,dry_seconds=duration,ref_seconds=reference_duration,
        dry_t=np.array(source),ref_t=np.array(reference_times),voiced=voiced,offset=0.,anchors=[],
        acoustic_trusted=evidence & voiced,correspondence_trusted=evidence & voiced)
    pieces,_ = plan_segments(plan,strength,max_rate)
    out,_ = render_segments(dry,pieces,sr,end=max(len(dry),len(reference)))
    stats = summarize(pieces,plan,max_rate)
    log(f"[节奏] 中文音素锚点 {stats['lyric_anchors']}，实际变时 {len(stats['active_intervals_seconds'])} 段")
    if capture is not None: capture.update(rhythm_plan=plan,rhythm_pieces=pieces)
    return out,stats
