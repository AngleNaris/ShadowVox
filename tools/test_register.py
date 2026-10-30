"""音区保护与力度的已知答案测试，不依赖真实演唱评分。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadowvox.register import fold_register, register_shift
from shadowvox.align import estimate_shift
from shadowvox.f0 import from_cent


class RegisterTests(unittest.TestCase):
    def test_both_octaves_preserved_and_large_error_corrected(self):
        ref = np.full(400, 5400.)
        for octave in (-2, -1, 0, 1):
            dry = ref + octave*1200 + 250
            previous = 0
            for strength in (.45, .75, 1.):
                shift, good, octaves = register_shift(dry, ref, np.ones(400, bool), strength)
                np.testing.assert_array_equal(octaves, octave)
                self.assertTrue(good.all())
                np.testing.assert_allclose(shift[50:-50], -250*strength)
                self.assertGreater(abs(float(np.median(shift))), previous)
                previous = abs(float(np.median(shift)))

    def test_phrase_and_sustained_register_change(self):
        for gap in (0, 40):
            ref = np.full(600+gap, 5400.)
            dry = np.r_[np.full(300, 4250.), np.zeros(gap), np.full(300, 5450.)]
            good = dry > 0
            shift, reliable, octaves = register_shift(dry, ref, good, 1)
            np.testing.assert_array_equal(octaves[30:270], -1)
            np.testing.assert_array_equal(octaves[330+gap:-30], 0)
            np.testing.assert_allclose(shift[50:250], -50)
            np.testing.assert_array_equal(shift[~good], 0)

    def test_short_octave_glitch_is_not_pulled(self):
        ref = np.full(400, 5400.)
        dry = ref-1200+50
        dry[180:195] += 1200
        shift, reliable, octaves = register_shift(dry, ref, np.ones(400, bool), 1)
        np.testing.assert_array_equal(octaves, -1)
        np.testing.assert_array_equal(shift[180:195], 0)
        self.assertFalse(reliable[180:195].any())

    def test_ambiguous_half_octave_is_not_corrected(self):
        ref = np.full(400, 5400.)
        dry = ref-600+20*np.sin(np.arange(400))
        shift, reliable, _ = register_shift(dry, ref, np.ones(400, bool), 1)
        np.testing.assert_array_equal(shift, 0)
        self.assertFalse(reliable.any())

    def test_strength_keeps_singer_vibrato_and_unvoiced(self):
        ref = np.full(500, 5400.)
        dry = ref-1200+200+30*np.sin(2*np.pi*5*np.arange(500)/100)
        voiced = np.ones(500, bool)
        voiced[200:240] = False
        for strength in (.45, .75, 1.):
            shift, _, _ = register_shift(dry, ref, voiced, strength)
            np.testing.assert_array_equal(shift[~voiced], 0)
            self.assertAlmostEqual(float(np.median(shift[50:150])), -200*strength, delta=3)
            # 中心被修正，演唱者 ±30c 颤音大部分保留（修正量振荡低于颤音峰峰值的 40%）
            self.assertGreater(float(np.ptp((dry+shift)[50:150])), 40)
            self.assertLess(float(np.ptp(shift[60:140])), .4*60*strength)

    def test_reference_wobble_not_transplanted_but_notes_follow(self):
        t = np.arange(600)/100
        ref = np.r_[np.full(300, 5400.), np.full(300, 5600.)]+25*np.sin(2*np.pi*6*t)
        dry = np.r_[np.full(300, 5450.), np.full(300, 5550.)]
        shift, _, _ = register_shift(dry, ref, np.ones(600, bool), 1)
        out = dry+shift
        self.assertAlmostEqual(float(np.median(out[50:250])), 5400, delta=5)
        self.assertAlmostEqual(float(np.median(out[350:550])), 5600, delta=5)
        self.assertLess(float(np.ptp(out[60:240])), 20)

    def test_alignment_integration_preserves_low_octave(self):
        ref = from_cent(np.full(500, 5400.))
        dry = from_cent(np.full(500, 4450.))
        result = estimate_shift(dry, ref, strength=1, offset_override=0, pitch_mode="register")
        np.testing.assert_allclose(result.shift_cents[50:-50], -250, atol=.01)
        self.assertEqual(result.stats["register_octave_counts"], {"-1": 500})


if __name__ == "__main__":
    unittest.main()
