"""修音边界的已知答案验收：保护区、连续衔接、音高与短尾完整性。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox.psola import _controls, render


class PsolaTransitionTests(unittest.TestCase):
    def test_silent_control_bypasses_every_sample(self):
        raw = np.random.default_rng(2).normal(0, .1, 23017).astype(np.float32)
        np.testing.assert_array_equal(render(raw, 22050, np.zeros(105)), raw)

    def test_pitch_entry_exit_and_middle_strength(self):
        shift = np.r_[np.zeros(20), np.full(80, 300.), np.zeros(20)]
        before = shift.copy()
        curve, blend = _controls(shift, 44100, 52920, 100)
        np.testing.assert_array_equal(shift, before)
        np.testing.assert_array_equal(curve[:20], 0)
        np.testing.assert_array_equal(curve[100:], 0)
        np.testing.assert_array_equal(curve[30:90], 300)
        self.assertLess(np.max(np.abs(np.diff(curve))), 90)
        self.assertLess(np.max(np.abs(np.diff(blend))), .001)

    def test_opposed_waveforms_have_no_frame_edge_click(self):
        # 固定相位差的最坏情况：旧的 1/3 混合跳档会产生 2/3 的波形台阶。
        _, blend = _controls(np.r_[np.zeros(20), np.full(60, 250.), np.zeros(20)],
                             44100, 44100, 100)
        mixed = 1. - 2.*blend
        self.assertLess(np.max(np.abs(np.diff(mixed))), .002)
        np.testing.assert_array_equal(mixed[:8820], 1)
        np.testing.assert_array_equal(mixed[35280:], 1)

    def test_real_synthesis_keeps_protected_gap_exact(self):
        sr = 22050
        t = np.arange(round(sr*1.537))/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        shift = np.full(154, 180.)
        shift[45:65] = 0
        out = render(raw, sr, shift)
        frame = np.minimum((np.arange(len(raw))*100/sr).astype(int), len(shift)-1)
        np.testing.assert_array_equal(out[shift[frame] == 0], raw[shift[frame] == 0])
        self.assertEqual(out.shape, raw.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_steady_note_still_receives_full_correction(self):
        sr = 22050
        t = np.arange(2*sr)/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        for cents in (-300., 300.):
            out = render(raw, sr, np.full(200, cents))
            middle = out[sr//2:3*sr//2]
            nfft = 262144
            spectrum = abs(np.fft.rfft(middle*np.hanning(len(middle)), n=nfft))
            measured = np.argmax(spectrum)*sr/nfft
            error = 1200*np.log2(measured/(220*2**(cents/1200)))
            self.assertLess(abs(error), 3.)

    def test_sub_frame_tail_and_short_active_regions_are_finite(self):
        curve, blend = _controls(np.array([200., 0., -200., 200.]), 44100, 1501, 100)
        self.assertTrue(np.isfinite(curve).all() and np.isfinite(blend).all())
        self.assertTrue(np.all((blend >= 0) & (blend <= 1)))
        np.testing.assert_array_equal(blend[441:882], 0)
        self.assertLess(float(blend[-1]), .001)

    def test_removed_single_frame_correction_does_not_resynthesize(self):
        raw = np.random.default_rng(3).normal(0, .1, 22050).astype(np.float32)
        shift = np.zeros(100)
        shift[50] = 200
        with patch('shadowvox.psola.parselmouth.Sound', side_effect=AssertionError('should bypass')):
            np.testing.assert_array_equal(render(raw, 22050, shift), raw)


if __name__ == '__main__':
    unittest.main()
