"""复用已分析素材，后台重算播放位置附近的草稿。"""
import numpy as np

from . import audio, config as C


class PreviewCache:
    def __init__(self, capture, report, preview):
        self.data = capture
        self.report = report
        self.initial = capture['options']
        self.preview_gain = 10 ** (preview['gains_db'][1] / 20)
        self.fixed = capture['dry_before_denoise'] if self.initial['declip'] else None
        self.cleaned = {}
        if 'denoised_dry' in capture:
            self.cleaned[self.initial['declip']] = capture['denoised_dry']
        self.cleaned_chunks = {}
        self.extractor = None
        self.full_shift = None
        self.shift_detail = None

    def source(self, opts, segment=None):
        from .repair import repair_clipping
        raw = self.data['dry_input']
        if opts.declip:
            if self.fixed is None:
                self.fixed = repair_clipping(raw, C.VOC_SR)[0]
            source = self.fixed
        else:
            source = raw
        if not opts.denoise or not opts.denoise_strength:
            return source
        if opts.declip not in self.cleaned:
            from .denoise import process
            if segment is None:
                self.cleaned[opts.declip] = process(source, C.VOC_SR)
            else:
                a = max(0, int(segment[0] - C.VOC_SR))
                b = min(len(source), int(segment[1] + C.VOC_SR))
                if b <= a:
                    return source
                key = (opts.declip, a, b)
                if key not in self.cleaned_chunks:
                    self.cleaned_chunks.clear()
                    self.cleaned_chunks[key] = process(source[a:b], C.VOC_SR)
                local = source.copy()
                local[a:b] += (self.cleaned_chunks[key] - source[a:b]) * opts.denoise_strength
                return local
        clean = self.cleaned[opts.declip]
        if opts.denoise_strength == 1:
            return clean
        return source + (clean - source) * opts.denoise_strength

    def plan(self, opts, source):
        from .full_align import align, plan_segments
        strength = opts.rhythm_strength if opts.rhythm else 0.
        plan = self.data.get('rhythm_plan')
        if plan is None:
            hint = 0.
            if opts.alignment_backend == 'hubertfa_zh':
                from .rhythm import onset_offset
                hint = -onset_offset(self.data['input_f0'], self.data['ref_f0'])
            align(source, self.data['reference_original'], C.VOC_SR, log=lambda _: None,
                  strength=0., capture=self.data, max_rate=opts.rhythm_max_rate,
                  backend=opts.alignment_backend, offset_hint=hint)
            plan = self.data['rhythm_plan']
        if plan.get('backend') == 'hubertfa_zh':
            from .hubert_alignment import plan_segments as chinese_plan
            pieces, stats = chinese_plan(plan, strength, opts.rhythm_max_rate)
            return pieces, stats, 0.
        pieces, stats = plan_segments(plan['dry_t'], plan['ref_t'], plan['dry_seconds'],
            plan['ref_seconds'], plan['voiced'], anchors=plan['anchors'], offset=plan['offset'],
            acoustic_trusted=plan['acoustic_trusted'], strength=strength,
            max_rate=opts.rhythm_max_rate)
        return pieces, stats, plan['offset']

    @staticmethod
    def _placed(source, start, end, move):
        sr = C.VOC_SR
        count = max(0, end - start)
        out = np.zeros(count, dtype=np.float32)
        shift = round(move * sr)
        src_start = max(0, start - shift)
        src_end = min(len(source), end - shift)
        if src_end > src_start:
            dest = src_start + shift - start
            out[dest:dest + src_end - src_start] = source[src_start:src_end]
        return out

    @staticmethod
    def _warped(source, pieces, start, end):
        from .full_align import render_segments
        return render_segments(source, pieces, C.VOC_SR, start=start, end=end)

    def _pitch(self, vocal, reference, opts, protected, start, mapping=None):
        if not opts.strength:
            return vocal
        from .f0 import FcpeExtractor
        from .align import estimate_shift
        from .psola import render
        if self.extractor is None:
            self.extractor = FcpeExtractor()
        dry_f0 = self.extractor.extract(vocal, C.VOC_SR)
        ref_f0 = self.extractor.extract(reference, C.VOC_SR)
        pitch_reference = {}
        if mapping is not None:
            from .full_align import pitch_mapping
            times, valid = pitch_mapping(*mapping, len(dry_f0), C.F0_FRAME_RATE, start/C.VOC_SR)
            pitch_reference = dict(reference_times=times-start/C.VOC_SR, reference_valid=valid)
            protected = []
        shift = estimate_shift(dry_f0, ref_f0, strength=opts.strength,
                               max_shift_cents=opts.max_shift_cents, offset_override=0., pitch_mode='register',
                               register_detail=opts.register_detail,
                               **pitch_reference).shift_cents
        for lo, hi in protected:
            a = max(0, round((lo-start/C.VOC_SR) * C.F0_FRAME_RATE))
            b = max(0, round((hi-start/C.VOC_SR) * C.F0_FRAME_RATE))
            shift[a:b] = 0
        return render(vocal, C.VOC_SR, shift)

    def render(self, opts, position, seconds=12, *, checkpoint=None):
        """返回窗口起点与立体声草稿，允许调用方在处理阶段之间取消。"""
        check = checkpoint if checkpoint is not None else lambda: None
        from .mixing import enhance, bus_compress, master
        sr = C.VOC_SR
        total = self.data['inst'].shape[1]
        start = max(0, min(total-1, round((position-2) * sr)))
        end = min(total, max(start + 1, round((position + seconds) * sr)))
        local_reference = None
        reference_shift = self.report.get('rhythm', {}).get('reference_shift_seconds',0.)
        same_source = all(getattr(opts, key) == self.initial[key] for key in
                          ('denoise', 'denoise_strength', 'declip', 'rhythm', 'rhythm_strength', 'rhythm_max_rate'))
        chinese_pitch = (opts.alignment_backend == 'hubertfa_zh' and opts.rhythm_mode == 'full'
                         and opts.pitch_mode == 'register' and opts.strength)
        if chinese_pitch and not self.data.get('pitch_reference'):
            same_source = False
        detail = (opts.register_detail, opts.max_shift_cents)
        original_detail = (self.initial.get('register_detail',0.),self.initial.get('max_shift_cents',450.))
        reused_vocal = (same_source and opts.strength == self.initial['strength'] and detail == original_detail
                        and opts.mute_reference_gaps == self.initial.get('mute_reference_gaps',False))
        if reused_vocal:
            vocal = self.data['vocal'][start:end].copy()
            vocal *= 10 ** ((opts.vocal_gain_db - self.initial['vocal_gain_db']) / 20)
            accompaniment = self.data['inst'][:, start:end].copy()
        elif same_source:
            from .align import estimate_shift
            from .psola import render
            if self.full_shift is None or self.shift_detail != detail:
                initial_strength = self.initial['strength']
                if initial_strength and detail == original_detail:
                    self.full_shift = self.data['shift_cents'] / initial_strength
                else:
                    pitch_reference = self.data.get('pitch_reference', {})
                    self.full_shift = estimate_shift(self.data['dry_f0'], self.data['ref_f0'],
                        strength=1., max_shift_cents=opts.max_shift_cents, offset_override=-self.data['move'],
                        pitch_mode='register', register_detail=opts.register_detail, **pitch_reference).shift_cents
                    protected = [] if pitch_reference else self.report.get('rhythm', {}).get('protected_intervals_seconds', [])
                    for lo, hi in protected:
                        self.full_shift[round(lo*C.F0_FRAME_RATE):round(hi*C.F0_FRAME_RATE)] = 0
                self.shift_detail = detail
            move = self.data['move']
            vocal = self._placed(self.data['aligned_dry'], start, end, move)
            times = (start / sr - move) + np.arange(int(np.ceil(len(vocal) / sr * C.F0_FRAME_RATE))) / C.F0_FRAME_RATE
            shift = np.interp(times * C.F0_FRAME_RATE, np.arange(len(self.full_shift)),
                              self.full_shift, left=0., right=0.) * opts.strength
            if opts.strength:
                vocal = render(vocal, sr, shift)
            vocal *= self.data['vocal_normalization'] * 10 ** (opts.vocal_gain_db / 20)
            accompaniment = self.data['inst'][:, start:end].copy()
        else:
            mapping = None
            if (opts.rhythm and opts.rhythm_strength) or chinese_pitch:
                raw = self.data['dry_before_denoise'] if self.initial['declip'] else self.data['dry_input']
                pieces, stats, _ = self.plan(opts, raw)
                check()
                reference_shift = stats['reference_shift_seconds']
                bounds = []
                for piece in pieces:
                    lo, hi = max(start/sr, piece['target'][0]), min(end/sr, piece['target'][-1])
                    if hi > lo:
                        bounds.extend(np.interp([lo, hi], piece['target'], piece['source']))
                segment = (min(bounds)*sr, max(bounds)*sr) if bounds else (0, 0)
                source = self.source(opts, segment=segment)
                check()
                vocal, protected = self._warped(source, pieces, start, end)
                mapping = (pieces, self.data['rhythm_plan'])
            else:
                move = 0.
                reference_shift = 0.
                source = self.source(opts, segment=(start-move*sr, end-move*sr))
                check()
                vocal = self._placed(source, start, end, move)
                protected = []
            reference = self._placed(self.data['reference_original'], start, end, reference_shift)
            local_reference = reference
            original_bed = self.data['inst_original']
            accompaniment = np.vstack([self._placed(ch, start, end, reference_shift) for ch in original_bed])
            vocal = self._pitch(vocal, reference, opts, protected, start, mapping)
            vocal *= self.data['vocal_normalization'] * 10 ** (opts.vocal_gain_db / 20)
        check()
        if opts.mute_reference_gaps and not reused_vocal:
            from .vocal_gate import mute_gaps
            vocal, _ = mute_gaps(vocal,self.data['reference_singing'],sr,start-round(reference_shift*sr))
        # 较短尾段补零，保证滤波器与包络计算有足够输入。
        length = len(vocal)
        if length < sr // 2:
            pad = sr // 2 - length
            vocal = np.pad(vocal, (0, pad))
            accompaniment = np.pad(accompaniment, ((0, 0), (0, pad)))
        baseline = self.report.get('mixing', {})
        if opts.mix_effects:
            curve = self.data.get('mix_gain_curve')
            gain_db = (np.interp((start + np.arange(len(vocal))) / sr,
                                curve['times'], curve['gain_db']) if curve is not None else None)
            reference = None
            if curve is None:
                if local_reference is not None:
                    reference = np.pad(local_reference, (0, max(0, len(vocal)-len(local_reference))))
                elif self.data.get('reference') is not None:
                    reference = self._placed(self.data['reference'], start, start+len(vocal), 0.)
            mixed, _ = enhance(vocal, accompaniment, sr, space=opts.mix_space,
                lowcut_hz=opts.mix_lowcut_hz, eq=opts.mix_eq, compression=opts.mix_compression,
                deessing=opts.mix_deess, harmonics=opts.mix_harmonics, reverb=opts.mix_reverb, delay_amount=opts.mix_delay,
                duck=opts.mix_duck, reference_vocal=reference,
                vocal_gain_db=opts.vocal_gain_db, gain_db=gain_db)
            check()
            if opts.fixed_master:
                if opts.master_density and baseline.get('input_lufs') is not None:
                    from .density import densify
                    mixed, _ = densify(mixed, sr, baseline['input_lufs'], opts.master_density,
                                       baseline.get('density'), checkpoint=checkpoint)
            elif opts.master_density is not None:
                mixed, _ = master(mixed,sr,opts.target_lufs,opts.true_peak_ceiling,
                                  density=opts.master_density,density_reference=baseline.get('density'))
            else:
                gain = baseline.get('master_gain_db', 0.) + opts.target_lufs - baseline.get('target_lufs', -16.)
                if baseline.get('bus_compression', {}).get('enabled'):
                    bus_gain = baseline.get('bus_normalization_db', 0.)
                    mixed, _ = bus_compress(mixed * 10**(bus_gain / 20), sr)
                    gain -= bus_gain
                gain += baseline.get('true_peak_correction_db', 0.)
                mixed *= 10 ** (gain / 20)
        else:
            mixed = accompaniment + vocal[None, :]
        if opts.fixed_master:
            # 草稿用整曲的母带增益近似，避免每次编辑重新迭代整曲响度。
            mixed *= 10**((baseline.get('master_gain_db', 0.)+baseline.get('true_peak_correction_db', 0.))/20.)
        peak = audio.peak(mixed)
        ceiling = 10 ** (opts.true_peak_ceiling / 20)
        if peak > ceiling:
            mixed *= ceiling / peak
        mixed *= self.preview_gain
        return start, np.ascontiguousarray(mixed[:, :length].T, dtype=np.float32)
