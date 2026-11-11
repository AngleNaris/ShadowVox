"""已知答案回归：偏移方向、共同网格、颤音和音符/清音边界。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadowvox.f0 import from_cent, cent
from shadowvox.align import estimate_shift, apply_shift
from shadowvox.natural import center_shift
from eval_quality import compare, on_reference_grid


class NaturalnessTests(unittest.TestCase):
    def test_positive_and_negative_delays(self):
        reference = from_cent(4000 + .5 * np.arange(1000))
        for frames in (20, -20):
            dry = (np.r_[np.zeros(frames), reference] if frames > 0
                   else reference[-frames:])
            result = compare(dry, reference, reference, frames / 100)
            self.assertLess(result["before_median"], .001)
            self.assertLess(result["after_median"], .001)
        self.assertEqual(on_reference_grid(np.ones(20), .1, 20)[-1], 0)

    def test_common_grid_excludes_unvoiced(self):
        ref = np.full(100, 220.)
        fixed = ref.copy()
        fixed[:25] = 0
        self.assertEqual(compare(ref, fixed, ref, 0)["common_frames"], 75)

    def test_vibrato_and_center(self):
        t = np.arange(1000) / 100
        base = float(cent(np.array([220.]))[0])
        dry = from_cent(base + 40 + 30 * np.sin(2*np.pi*5*t))
        ref = np.full(len(t), 220., dtype=np.float32)
        output = {}
        for mode in ("frame", "natural"):
            r = estimate_shift(dry, ref, strength=.85, offset_override=0, pitch_mode=mode)
            output[mode] = cent(apply_shift(dry, r.shift_cents))[100:-100] - base
        self.assertAlmostEqual(float(np.ptp(output["natural"]) / 2), 30, places=2)
        self.assertAlmostEqual(float(np.median(output["natural"])), 6, places=2)
        self.assertLess(float(np.ptp(output["frame"]) / 2), 5)

    def test_note_change_and_unvoiced_gap(self):
        target = np.r_[np.full(100, 5400.), np.full(100, 5600.), np.zeros(20)]
        dry = target + np.r_[np.full(100, 40.), np.full(100, -40.), np.zeros(20)]
        shift = center_shift(dry, target, target > 0)
        np.testing.assert_allclose(shift[20:80], -34)
        np.testing.assert_allclose(shift[120:180], 34)
        np.testing.assert_array_equal(shift[200:], 0)

    def test_short_note_passthrough(self):
        shift = center_shift(np.full(9, 5400.), np.full(9, 5440.), np.ones(9, bool))
        np.testing.assert_array_equal(shift, 0)

    def test_signalsmith_dynamic_pitch_and_duration(self):
        from scipy.signal import hilbert
        from shadowvox.config import VOC_SR as SR
        from shadowvox.signalsmith import render_pitch
        t = np.arange(5 * SR) / SR
        x = (.2 * np.sin(2*np.pi*220*t)).astype(np.float32)
        shift = np.zeros(500)
        shift[:8] = 400.
        shift[200:206] = 400.
        shift[-8:] = 400.
        y = render_pitch(x, shift, SR, f0=np.array([220.]))
        self.assertEqual(len(y), len(x))
        instantaneous = np.diff(np.unwrap(np.angle(hilbert(y.astype(np.float64))))) * SR / (2*np.pi)
        for center, expected in ((.04, 400.), (1., 0.), (2.03, 400.), (4.96, 400.)):
            with self.subTest(center=center):
                frequency = np.median(instantaneous[round((center-.004)*SR):round((center+.004)*SR)])
                self.assertAlmostEqual(1200*np.log2(frequency/220), expected, delta=8.)
        untouched = render_pitch(x, np.zeros_like(shift), SR)
        np.testing.assert_array_equal(untouched, x)

    def test_signalsmith_timing_follows_the_map(self):
        from scipy.ndimage import gaussian_filter1d
        from shadowvox.config import VOC_SR as sr
        from shadowvox.signalsmith import stretch
        audio = np.zeros(3 * sr, dtype=np.float32)
        source = np.array([0., .8, 1.6, 2.4, 3.])
        target = np.array([0., .87, 1.6, 2.34, 3.03])
        times = np.array([.45, 1.25, 2., 2.75])
        lag = np.arange(-380, 381)
        noise = np.random.default_rng(123)
        for sample in np.rint(times*sr).astype(int):
            audio[sample+lag] = noise.normal(size=len(lag))*np.exp(-.5*(lag/120)**2)
        rendered = stretch(audio, source, target, sr)
        self.assertEqual(len(rendered), round(target[-1]*sr))
        energy = gaussian_filter1d(rendered**2, sr*.002)
        for expected in np.interp(times, source, target):
            lo, hi = round((expected-.08)*sr), round((expected+.08)*sr)
            actual = (lo + np.argmax(energy[lo:hi])) / sr
            self.assertAlmostEqual(actual, expected, delta=.003)
        np.testing.assert_array_equal(stretch(audio, source, source, sr), audio)


if __name__ == "__main__":
    unittest.main()
