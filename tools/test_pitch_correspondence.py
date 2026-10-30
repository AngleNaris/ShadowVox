"""音符对应独立于变时；使用已知音符/时标检验错误参考与回退。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox import full_align as fa
from shadowvox.align import estimate_shift
from shadowvox.live_preview import PreviewCache
from shadowvox.pipeline import Options


def fixture():
    source = np.arange(0., 4.01, .1)
    pieces = [dict(source=source, target=source.copy(), protected=True,
                   render_fallback='opposed_boundary_phase')]
    plan = dict(dry_t=source, ref_t=source+.2, offset=0., ref_seconds=5.,
                acoustic_trusted=np.zeros(40, bool), correspondence_trusted=np.ones(40, bool))
    return pieces, plan


class PitchCorrespondenceTests(unittest.TestCase):
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
        cache = PreviewCache({'options': {'declip': False}}, {}, {'gains_db': [0, 0]})
        cache.extractor = SimpleNamespace(extract=lambda *_: np.full(401, 220.))
        with patch('shadowvox.live_preview.C.VOC_SR', 100), \
             patch('shadowvox.align.estimate_shift', return_value=SimpleNamespace(shift_cents=np.ones(401))) as estimate, \
             patch('shadowvox.psola.render', side_effect=lambda vocal, sr, shift: shift):
            shift = cache._pitch(np.ones(400), np.ones(400), Options(), [(0., 4.)], 0,
                                 (pieces, plan))
        self.assertTrue(estimate.call_args.kwargs['reference_valid'][:400].all())
        np.testing.assert_array_equal(shift, 1.)


if __name__ == '__main__':
    unittest.main()
