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

    def test_rubberband_dynamic_pitch_and_duration(self):
        from naturalness_lab import rubberband, SR
        t = np.arange(5 * SR) / SR
        x = (.2 * np.sin(2*np.pi*220*t)).astype(np.float32)
        y, delta = rubberband(x, np.r_[np.zeros(200), np.full(300, 100.)])
        self.assertEqual(len(y), len(x))
        self.assertLess(abs(delta), SR * .01)
        for start, expected in ((.5, 220.), (3., 220 * 2**(1/12))):
            cut = y[int(start*SR):int((start+1)*SR)]
            peak_hz = np.argmax(np.abs(np.fft.rfft(cut*np.hanning(len(cut)))))
            self.assertLess(abs(peak_hz-expected), 1.5)


if __name__ == "__main__":
    unittest.main()
