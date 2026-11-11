"""音符对应独立于变时；使用已知音符/时标检验错误参考与回退。"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox import full_align as fa
from shadowvox.align import estimate_shift
from shadowvox.f0 import from_cent
from shadowvox.live_preview import PreviewCache
from shadowvox.pipeline import Options


def fixture():
    source = np.arange(0., 4.01, .1)
    pieces = [dict(source=source, target=source.copy(), protected=True,
                   render_fallback='opposed_boundary_phase')]
    plan = dict(dry_t=source, ref_t=source+.2, offset=0., ref_seconds=5.,
                acoustic_trusted=np.zeros(40, bool), correspondence_trusted=np.ones(40, bool))
    return pieces, plan


def note_result(dry, ref, *, valid=None, times=None, **kwargs):
    size = len(dry)
    options = dict(strength=1., max_shift_cents=450., pitch_mode='register', offset_override=0.,
                   reference_times=np.arange(size)/100. if times is None else times,
                   reference_valid=np.ones(size, bool) if valid is None else valid)
    return estimate_shift(from_cent(dry), from_cent(ref), **(options | kwargs))


class PitchCorrespondenceTests(unittest.TestCase):
    def test_large_notes_are_skipped_and_normal_shifts_follow_strength(self):
        target = np.full(200, 5400.)
        for error in (-500., -301., 301., 500.):
            with self.subTest(error=error):
                source = target + error
                full = note_result(source, target, max_shift_cents=200.)
                np.testing.assert_array_equal(full.shift_cents, 0.)
                np.testing.assert_array_equal(full.confidence, 0.)
                coverage = full.stats['register_coverage']
                self.assertEqual(coverage['large_note_protected_notes'], 1)
                self.assertEqual(coverage['large_note_protected_frames'], 200)
                self.assertEqual(coverage['limited_frames'], 0)
        ordinary = note_result(target + 250., target, max_shift_cents=200.)
        half = note_result(target + 250., target, max_shift_cents=200., strength=.5)
        np.testing.assert_allclose(ordinary.shift_cents[5:-5], -200.)
        np.testing.assert_allclose(half.shift_cents, ordinary.shift_cents*.5)
        self.assertEqual(ordinary.stats['register_coverage']['limited_frames'], 200)
        boundary = note_result(target + 300., target)
        np.testing.assert_allclose(boundary.shift_cents[5:-5], -300., atol=.1)

    def test_octave_is_folded_before_testing_note_size(self):
        target = np.repeat([5600., 5900., 5800., 6100.], 90)
        for octave in (-1, 1):
            with self.subTest(octave=octave):
                source = target + 1200*octave
                exact = note_result(source, target)
                np.testing.assert_allclose(exact.shift_cents, 0., atol=.002)
                self.assertEqual(exact.stats['register_octave_counts'], {str(octave): len(source)})
                source += 55.
                source[180:270] += 320.
                corrected = note_result(source, target)
                np.testing.assert_allclose(corrected.shift_cents[20:160], -55., atol=.1)
                np.testing.assert_array_equal(corrected.shift_cents[190:260], 0.)
                np.testing.assert_allclose(corrected.shift_cents[285:345], -55., atol=.1)
                self.assertEqual(corrected.stats['register_coverage']['large_note_protected_notes'], 1)

    def test_isolated_ninety_ms_note_is_eligible(self):
        target = np.full(100, 5400.)
        valid = np.zeros(100, bool)
        valid[43:52] = True
        result = note_result(target - 210, target, valid=valid)
        np.testing.assert_allclose(result.shift_cents[46:49], 210., atol=.1)
        np.testing.assert_array_equal(result.shift_cents[~valid], 0.)
        coverage = result.stats['register_coverage']
        self.assertEqual(coverage['matched_voiced_frames'], 9)
        self.assertEqual(coverage['planned_frames'], 9)
        self.assertEqual(coverage['short_planned_frames'], 9)
        self.assertEqual(coverage['unmatched_reference_frames'], 91)

    def test_short_wrong_note_does_not_shift_neighboring_in_tune_notes(self):
        target = np.full(220, 5400.)
        source = target.copy()
        source[90:100] += 280.
        result = note_result(source, target)
        np.testing.assert_allclose(result.shift_cents[93:97], -280., atol=.1)
        np.testing.assert_array_equal(result.shift_cents[:90], 0.)
        np.testing.assert_array_equal(result.shift_cents[100:], 0.)

    def test_vibrato_keeps_its_shape_inside_a_supported_note(self):
        t = np.arange(220)/100.
        source = 5250. + 38*np.sin(2*np.pi*5*t)
        target = 5400. + 33*np.sin(2*np.pi*5*t + 1.6)
        result = note_result(source, target)
        interior = result.shift_cents[10:-10]
        source_shape = source[10:-10]
        corrected = source_shape + interior
        self.assertGreater(np.ptp(corrected), .6*np.ptp(source_shape))
        self.assertGreater(np.corrcoef(corrected, source_shape)[0, 1], .8)
        self.assertAlmostEqual(float(np.median(interior)), 150., delta=4.)

    def test_octave_glitch_and_ambiguous_register_remain_protected(self):
        target = np.full(400, 5400.)
        source = target - 1150.
        source[180:195] += 1200.
        result = note_result(source, target)
        np.testing.assert_array_equal(result.shift_cents[180:195], 0.)
        self.assertEqual(result.stats['register_coverage']['register_conflict_frames'], 15)
        self.assertGreater(np.abs(result.shift_cents[100]), 40.)
        ambiguous = note_result(target - 600., target)
        np.testing.assert_array_equal(ambiguous.shift_cents, 0.)

    def test_invalid_times_do_not_supply_pitch_and_skips_are_accounted_for(self):
        target = np.full(100, 5400.)
        times = np.arange(100)/100.
        times[8] = np.nan
        times[24:30] = .02
        valid = np.ones(100, bool)
        valid[30:40] = False
        result = note_result(target - 80., target, times=times, valid=valid)
        np.testing.assert_array_equal(result.shift_cents[24:40], 0.)
        self.assertEqual(result.shift_cents[8], 0.)
        coverage = result.stats['register_coverage']
        json.dumps(coverage)
        self.assertEqual(coverage['unmatched_reference_frames'], 17)
        self.assertEqual(coverage['matched_voiced_frames'],
                         sum(coverage[key] for key in ('planned_frames', 'large_note_protected_frames', 'short_frames',
                                                       'unstable_frames', 'register_conflict_frames')))

    def test_render_fallback_still_corrects_the_corresponding_note(self):
        pieces, plan = fixture()
        times, valid = fa.pitch_mapping(pieces, plan, 401)
        # 源 2 秒时换音；原唱晚 200ms，同刻查音会拿到前一个音。
        dry = np.r_[np.full(200, 220.), np.full(201, 330.)] * 2**(-80/1200)
        ref = np.r_[np.full(220, 220.), np.full(281, 330.)]
        result = estimate_shift(dry, ref, strength=1., pitch_mode='register', offset_override=0.,
                                reference_times=times, reference_valid=valid)
        np.testing.assert_allclose(result.shift_cents[205:215], 80., atol=.1)
        self.assertGreater(np.median(result.shift_cents[:390]), 79.)
        self.assertEqual(result.shift_cents[-1], 0.)

    def test_untrusted_gap_is_zero_even_with_matching_pitch(self):
        pieces, plan = fixture()
        plan['correspondence_trusted'][10:20] = False
        times, valid = fa.pitch_mapping(pieces, plan, 401)
        result = estimate_shift(np.full(401, 220.), np.full(501, 230.), pitch_mode='register',
                                offset_override=0., reference_times=times, reference_valid=valid)
        np.testing.assert_array_equal(result.shift_cents[100:200], 0.)
        np.testing.assert_array_equal(result.confidence[100:200], 0.)
        self.assertGreater(np.median(result.shift_cents[30:80]), 50.)

    def test_actual_warp_and_delayed_reference_share_time_origin(self):
        pieces, plan = fixture()
        plan.update(offset=-.3, ref_t=plan['dry_t']-.3)
        pieces[0]['target'] += .05*np.sin(np.pi*pieces[0]['source']/4)
        full, valid = fa.pitch_mapping(pieces, plan, 401)
        short, short_valid = fa.pitch_mapping(pieces, plan, 100, start_seconds=1.)
        np.testing.assert_allclose(short, full[100:200])
        np.testing.assert_array_equal(short_valid, valid[100:200])
        self.assertAlmostEqual(full[200], np.interp(2., pieces[0]['target'], pieces[0]['source']))

    def test_no_extrapolation_into_prefix_or_tail(self):
        pieces, plan = fixture()
        pieces[0]['target'] += .4
        _, valid = fa.pitch_mapping(pieces, plan, 501)
        self.assertFalse(valid[:40].any())
        self.assertFalse(valid[440:].any())
        self.assertTrue(valid[50:430].all())

    def test_correspondence_can_pass_when_rhythm_shift_is_too_large(self):
        source = np.arange(0., 4.01, .1)
        c = np.ones((12, 251))
        anchors = [(0., .25), (4., 4.25)]
        args = (c, c, source, source+.25, np.ones(40, bool), anchors)
        correspondence = fa._correspondence_support(*args)
        rhythm = fa._acoustic_support(*args)
        self.assertTrue(correspondence[5:35].all())
        self.assertFalse(rhythm.any())

    def test_preview_uses_independent_evidence_after_fallback(self):
        pieces, plan = fixture()
        cache = PreviewCache({'options': {'declip': False}}, {})
        cache.extractor = SimpleNamespace(extract=lambda *_: np.full(401, 220.))
        with patch('shadowvox.live_preview.C.VOC_SR', 100), \
             patch('shadowvox.align.estimate_shift', return_value=SimpleNamespace(shift_cents=np.ones(401))) as estimate, \
             patch('shadowvox.signalsmith.render_pitch', side_effect=lambda vocal, shift, sr, **_: shift) as render:
            shift = cache._pitch(np.ones(400), np.ones(400), Options(), [(0., 4.)], 0,
                                 (pieces, plan), renderer='signalsmith')
        render.assert_called_once()
        self.assertTrue(estimate.call_args.kwargs['reference_valid'][:400].all())
        np.testing.assert_array_equal(shift, 1.)


if __name__ == '__main__':
    unittest.main()
