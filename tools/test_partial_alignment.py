"""保序、连续性、错误锚点回退与实际 R3/PSOLA 的回归。"""
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox import full_align as fa
from shadowvox.live_preview import PreviewCache
from shadowvox.psola import render as pitch_render
from shadowvox.pipeline import Options


def grid(seconds=10.):
    return np.r_[np.arange(0., seconds, .1), seconds]


class ConservativeAlignmentTests(unittest.TestCase):
    def assert_safe(self, pieces, seconds, rate=.1):
        self.assertEqual(pieces[0]['source'][0], 0.)
        self.assertAlmostEqual(pieces[-1]['source'][-1], seconds)
        for piece in pieces:
            self.assertTrue(np.all(np.diff(piece['source']) > 0))
            ratios = np.diff(piece['target']) / np.diff(piece['source'])
            self.assertGreaterEqual(ratios.min(), 1-rate-1e-8)
            self.assertLessEqual(ratios.max(), 1+rate+1e-8)
            if piece['protected']:
                np.testing.assert_allclose(piece['target']-piece['source'],
                                           piece['target'][0]-piece['source'][0], atol=1e-8)
        for a, b in zip(pieces, pieces[1:]):
            self.assertEqual(a['source'][-1], b['source'][0])
            self.assertEqual(a['target'][-1], b['target'][0])

    def test_path_gap_does_not_cut_continuous_singing(self):
        for sign in (-1, 1):
            src = np.array([0., 2., 4., 4.02, 6., 10.])
            dst = src + np.array([0., 0., 0., .6, .6, .6])*sign
            pieces, stats = fa.plan_segments(src, dst, 10., 10., np.ones(100, bool),
                                             acoustic_trusted=np.ones(100, bool))
            self.assert_safe(pieces, 10.)
            self.assertAlmostEqual(pieces[-1]['target'][-1], 10.)
            self.assertEqual(stats['max_applied_shift_ms'], 0.)

    def test_uncertain_region_keeps_position_and_exact_samples(self):
        sr = 16000
        src = grid()
        dst = src + .1*np.sin(np.pi*src/10)
        evidence = np.ones(100, bool)
        evidence[40:60] = False
        pieces, _ = fa.plan_segments(src, dst, 10., 10., np.ones(100, bool), acoustic_trusted=evidence)
        self.assert_safe(pieces, 10.)
        raw = np.random.default_rng(5).uniform(.1, .3, 10*sr).astype(np.float32)
        with patch.object(fa, 'stretch', side_effect=lambda x, *_: x.copy()):
            out, _ = fa.render_segments(raw, pieces, sr)
        np.testing.assert_array_equal(out[4*sr:6*sr], raw[4*sr:6*sr])
        self.assertGreater(out.min(), .09)  # 保护切换处不能淡出到零

    def test_no_words_or_failed_recognition_is_bit_exact(self):
        sr = 16000
        raw = np.random.default_rng(1).normal(0, .1, 5*sr).astype(np.float32)
        c = np.ones((12, 250))
        with patch.object(fa, 'features', return_value=(c, c)), \
             patch.object(fa, '_warping_path', return_value=(grid(5), grid(5)*1.8)), \
             patch('shadowvox.lyric_anchors.find_anchors', side_effect=RuntimeError('unavailable')):
            out, stats = fa.align(raw, raw, sr, log=lambda _: None)
        np.testing.assert_array_equal(out, raw)
        self.assertFalse(stats['applied'])
        self.assertEqual(stats['alignment_basis'], 'preserved')

    def test_far_opening_and_tail_anchors_do_not_move_neighboring_lyrics(self):
        src = grid(40.)
        anchors = [(8., .1), (9., 1.1), (10., 2.1), (20., 20.), (23., 23.),
                   (26., 26.), (29., 29.), (32., 32.), (35., 36.3), (38., 39.3)]
        self.assertEqual(fa._global_offset(anchors, src, src), (0., False))
        accepted = fa._local_anchors(anchors, 0.)
        self.assertTrue(all(20 <= a <= 32 for a, _ in accepted))
        pieces, _ = fa.plan_segments(src, src+.1, 40., 40., np.ones(400, bool),
                                    anchors=anchors, acoustic_trusted=np.ones(400, bool))
        self.assert_safe(pieces, 40.)
        for piece in pieces:
            if piece['source'][0] < 20. or piece['source'][-1] > 32.:
                self.assertTrue(piece['protected'])

    def test_global_delay_requires_spread_out_agreement_and_keeps_prefix(self):
        src = grid(40.)
        anchors = [(a, a-.35) for a in (8., 12., 20., 28., 36.)]
        offset, trusted = fa._global_offset(anchors, src, src-.35)
        self.assertTrue(trusted)
        self.assertAlmostEqual(offset, -.35)
        self.assertEqual(fa._global_offset(anchors[:3], src, src-.35), (0., False))
        pieces, stats = fa.plan_segments(src, src-.35, 40., 40., np.ones(400, bool),
                                         anchors=anchors, offset=-.35, acoustic_trusted=np.ones(400, bool))
        self.assert_safe(pieces, 40.)
        self.assertAlmostEqual(stats['reference_shift_seconds'], .35)
        sr = 1000
        raw = np.arange(40*sr, dtype=np.float32)
        out, _ = fa.render_segments(raw, pieces, sr)
        np.testing.assert_array_equal(out, raw)

    def test_crossing_anchors_are_rejected_instead_of_sorted(self):
        anchors = [(1., 2.), (2., 1.), (3., 3.)]
        self.assertEqual(fa._local_anchors(anchors, 0.), [])
        self.assertEqual(fa._global_offset(anchors, grid(), grid()), (0., False))
        with self.assertRaises(ValueError):
            fa.stretch(np.ones(100), np.array([0., 1., 2.]), np.array([0., 2., 1.]), 100)

    def test_random_paths_never_break_order_coverage_or_rate(self):
        rng = np.random.default_rng(7)
        src = grid()
        for _ in range(25):
            candidate = np.r_[0., np.cumsum(rng.uniform(.03, .24, 100))]
            evidence = rng.uniform(size=100) > .1
            pieces, _ = fa.plan_segments(src, candidate, 10., 20., np.ones(100, bool),
                                         acoustic_trusted=evidence, max_rate=.03)
            self.assert_safe(pieces, 10., .03)

    def test_strength_and_rate_are_effective(self):
        src = grid()
        dst = src+.12*np.sin(np.pi*src/10)
        args = (src, dst, 10., 10., np.ones(100, bool))
        evidence = np.ones(100, bool)
        full, _ = fa.plan_segments(*args, acoustic_trusted=evidence, max_rate=.02)
        half, _ = fa.plan_segments(*args, acoustic_trusted=evidence, max_rate=.02, strength=.5)
        self.assert_safe(full, 10., .02)
        for a, b in zip(full, half):
            np.testing.assert_allclose(b['target']-b['source'], (a['target']-a['source'])*.5, atol=1e-8)
        self.assertGreater(np.max(full[0]['target']-full[0]['source']), .01)

    def test_real_stretcher_preserves_tone_order_and_has_no_added_gaps(self):
        sr = 16000
        t = np.arange(8*sr)/sr
        raw = (.2*np.sin(2*np.pi*np.where(t<4, 220*t, 220*4+440*(t-4)))).astype(np.float32)
        src = grid(8.)
        pieces, _ = fa.plan_segments(src, src+.03*np.sin(np.pi*src/8), 8., 8.,
                                     np.ones(80, bool), acoustic_trusted=np.ones(80, bool))
        out, _ = fa.render_segments(raw, pieces, sr)
        self.assertFalse(pieces[0]['protected'])
        self.assertGreater(np.max(abs(out-raw)), .01)
        self.assertEqual(out.shape, raw.shape)
        rms = np.sqrt(np.mean(out.reshape(-1, 160)**2, axis=1))
        self.assertGreater(rms.min(), .05)
        for a, b, hz in ((1, 3, 220), (5, 7, 440)):
            power = np.abs(np.fft.rfft(out[a*sr:b*sr]))
            peak = np.argmax(power)*sr/(sr*(b-a))
            self.assertAlmostEqual(peak, hz, delta=2.)

    def test_preview_preserves_protected_waveform_and_continuity(self):
        sr = 44100
        raw = np.full(4*sr, .25, dtype=np.float32)
        src = grid(4.)
        pieces, _ = fa.plan_segments(src, src+.5, 4., 4., np.ones(40, bool), anchors=[])
        full, _ = fa.render_segments(raw, pieces, sr)
        part, _ = PreviewCache._warped(raw, pieces, sr, 3*sr)
        np.testing.assert_array_equal(part, full[sr:3*sr])
        np.testing.assert_array_equal(part, raw[sr:3*sr])

    def test_opposed_boundary_phase_falls_back_to_original(self):
        sr = 16000
        t = np.arange(4*sr)/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        src = grid(4.)
        pieces, _ = fa.plan_segments(src, src+.1*np.sin(np.pi*src/4), 4., 4.,
                                     np.ones(40, bool), acoustic_trusted=np.ones(40, bool))
        with patch.object(fa, 'stretch', side_effect=lambda x, *_: -x):
            out, protected = fa.render_segments(raw, pieces, sr)
        np.testing.assert_array_equal(out, raw)
        self.assertTrue(protected)
        self.assertTrue(pieces[0]['protected'])
        self.assertEqual(pieces[0]['render_fallback'], 'opposed_boundary_phase')

    def test_psola_bypasses_zero_and_uncertain_regions_exactly(self):
        sr = 16000
        raw = (.2*np.sin(2*np.pi*220*np.arange(4*sr)/sr)).astype(np.float32)
        shift = np.zeros(401)
        np.testing.assert_array_equal(pitch_render(raw, sr, shift), raw)
        shift[100:200] = 80.
        out = pitch_render(raw, sr, shift)
        np.testing.assert_array_equal(out[:sr], raw[:sr])
        np.testing.assert_array_equal(out[2*sr:], raw[2*sr:])
        self.assertGreater(np.max(abs(out[sr:2*sr]-raw[sr:2*sr])), .01)

    def test_disabling_rhythm_does_not_reuse_an_old_global_move(self):
        sr = 44100
        raw = (.1*np.sin(2*np.pi*220*np.arange(4*sr)/sr)).astype(np.float32)
        opts = Options(rhythm_mode='full', strength=0, declip=False, denoise=False, mix_effects=False)
        bed = np.zeros((2,len(raw)), dtype=np.float32)
        capture = dict(options=asdict(opts), dry_input=raw, inst=bed, inst_original=bed,
                       reference_original=raw, vocal_normalization=1., move=0., rhythm_plan={'offset':-.35})
        cache = PreviewCache(capture, {}, {'gains_db':[0,0]})
        start, out = cache.render(replace(opts,rhythm=False), 0, seconds=4)
        self.assertEqual(start, 0)
        np.testing.assert_array_equal(out[:,0], raw)


if __name__ == '__main__':
    unittest.main()
