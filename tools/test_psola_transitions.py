"""修音边界的已知答案验收：保护区、连续衔接、音高与短尾完整性。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shadowvox.psola import _phase_lag, _pitch_changes, _transition, render


class PsolaTransitionTests(unittest.TestCase):
    def test_silent_control_bypasses_every_sample(self):
        raw = np.random.default_rng(2).normal(0, .1, 23017).astype(np.float32)
        np.testing.assert_array_equal(render(raw, 22050, np.zeros(105)), raw)

    def test_short_note_reaches_target_after_unshifted_crossfade(self):
        t = np.arange(44100)/44100.
        shift = np.full_like(t, 300.)
        short = _transition(.20, .26, 220., 220.)
        long = _transition(.20, .90, 220., 220.)
        curve = _pitch_changes(t, shift, [short])
        self.assertLess(short.glide, long.glide)
        np.testing.assert_array_equal(curve[(t <= short.start+short.fade_in) | (t >= short.end-short.fade_out)], 0.)
        np.testing.assert_array_equal(curve[(t >= .225) & (t <= .235)], 300.)
        np.testing.assert_array_equal(shift, 300.)

    def test_seam_detects_opposed_periods(self):
        period = 200
        raw = np.sin(2*np.pi*np.arange(44100)/period)
        self.assertEqual(abs(_phase_lag(raw, -raw, 8820, period)), period//2)
        self.assertEqual(_phase_lag(raw, raw, 8820, period), 0)

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

    def test_resynthesis_preserves_unvoiced_audio_between_pulse_runs(self):
        sr = 22050
        t = np.arange(2*sr)/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        raw[round(.75*sr):round(1.25*sr)] = np.random.default_rng(4).normal(0, .012, round(1.25*sr)-round(.75*sr))
        out = render(raw, sr, np.full(201, 350.))
        np.testing.assert_array_equal(out[round(.8*sr):round(1.2*sr)], raw[round(.8*sr):round(1.2*sr)])
        self.assertEqual(out.shape, raw.shape)

    def test_real_short_note_reaches_corrected_pitch(self):
        sr = 44100
        t = np.arange(2*sr)/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        shift = np.zeros(200)
        shift[100:106] = 400.
        out = render(raw, sr, shift)
        frequency = np.diff(np.unwrap(np.angle(hilbert(out.astype(np.float64))))) * sr/(2*np.pi)
        measured = np.median(frequency[round(1.025*sr):round(1.035*sr)])
        self.assertLess(abs(1200*np.log2(measured/(220*2**(400/1200)))), 90.)

    def test_real_pitch_exit_has_no_deep_envelope_cancellation_or_click(self):
        sr = 44100
        t = np.arange(3*sr)/sr
        raw = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        shift = np.zeros(300)
        shift[100:200] = 400.
        out = render(raw, sr, shift)
        envelope = np.abs(hilbert(out.astype(np.float64)))[round(1.95*sr):2*sr] / .2
        self.assertGreater(np.percentile(envelope, 5), .60)
        self.assertLess(np.max(np.abs(np.diff(out[2*sr-20:2*sr+20]))), .009)
        np.testing.assert_array_equal(out[2*sr:], raw[2*sr:])

    def test_removed_single_frame_correction_does_not_resynthesize(self):
        raw = np.random.default_rng(3).normal(0, .1, 22050).astype(np.float32)
        shift = np.zeros(100)
        shift[50] = 200
        with patch('shadowvox.psola.parselmouth.Sound', side_effect=AssertionError('should bypass')):
            np.testing.assert_array_equal(render(raw, 22050, shift), raw)


if __name__ == '__main__':
    unittest.main()
