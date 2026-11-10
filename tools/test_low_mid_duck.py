"""低中频遮蔽判断、旁路及带外保护。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shadowvox import mixing


class LowMidDuckTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.t = np.arange(5*self.sr) / self.sr
        self.body = np.sin(2*np.pi*200*self.t).astype(np.float32)
        self.bass = np.sin(2*np.pi*60*self.t).astype(np.float32)
        self.bed = np.stack([.24*self.body + .1*self.bass, -.24*self.body - .1*self.bass])

    def test_bypass_silence_and_unmasked_vocal(self):
        for voice, amount in ((.08*self.body, 0), (np.zeros_like(self.body), 1),
                              (.48*self.body, 1)):
            with self.subTest(amount=amount, vocal_peak=np.max(voice)):
                out, max_db = mixing._low_mid_duck(self.bed, voice, self.sr, amount)
                self.assertEqual(max_db, 0)
                np.testing.assert_array_equal(out, self.bed)

    def test_ducks_masked_band_in_both_channels_and_releases(self):
        voice = .08*self.body*((self.t >= 1) & (self.t < 3))
        out, max_db = mixing._low_mid_duck(self.bed, voice, self.sr, 1)
        self.assertAlmostEqual(max_db, 2.)
        np.testing.assert_allclose(out[0], -out[1], atol=1e-7)
        for quiet in (self.t < .8, self.t > 4.5):
            np.testing.assert_array_equal(out[:, quiet], self.bed[:, quiet])

        active = (self.t >= 1.5) & (self.t < 2.5)
        def attenuation(tone):
            return 20*np.log10(np.dot(out[0, active], tone[active]) /
                               np.dot(self.bed[0, active], tone[active]))

        self.assertAlmostEqual(attenuation(self.body), -2., delta=.12)
        self.assertAlmostEqual(attenuation(self.bass), 0., delta=.05)

    def test_single_vocal_reaches_mix_without_added_tracks(self):
        voice = .16*self.body
        bed = np.zeros_like(self.bed)
        expected, _ = mixing._vocal_chain(voice, self.sr, 75., 1., 1., 1., True)
        out, report = mixing.enhance(voice, bed, self.sr, space=0, gain_db=-2.)
        np.testing.assert_array_equal(out, np.stack([expected]*2)*np.float32(10**(-2/20)))
        self.assertEqual(report['max_low_mid_duck_db'], 0.)


if __name__ == '__main__':
    unittest.main()
