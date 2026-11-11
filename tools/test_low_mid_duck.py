"""伴奏中频避让的减益、旁路及带外保护。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shadowvox import mixing


class MidrangeDuckTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.t = np.arange(5*self.sr) / self.sr
        self.mid, self.bass, self.treble = [np.sin(2*np.pi*f*self.t).astype(np.float32)
                                            for f in (1000, 200, 5000)]
        channel = .24*self.mid + .1*self.bass + .1*self.treble
        self.bed = np.stack([channel, -channel])

    def test_bypass_amount_and_silence(self):
        for voice, amount in ((.25*self.mid, 0), (np.zeros_like(self.mid), 1)):
            with self.subTest(amount=amount, vocal_peak=np.max(voice)):
                out, _ = mixing._backing(self.bed, voice, self.sr, amount)
                np.testing.assert_array_equal(out, self.bed)

    def test_ducks_masked_band_in_both_channels_and_releases(self):
        voice = .35*self.mid*((self.t >= 1) & (self.t < 3))
        out, activity = mixing._backing(self.bed, voice, self.sr, 1)
        self.assertAlmostEqual(activity.max()*mixing.MID_DUCK_MAX_DB, 2.5*.85)
        np.testing.assert_allclose(out[0], -out[1], atol=1e-7)
        for quiet in (self.t < .8, self.t > 4.5):
            np.testing.assert_array_equal(out[:, quiet], self.bed[:, quiet])

        active = (self.t >= 1.5) & (self.t < 2.5)
        def attenuation(tone):
            return 20*np.log10(np.dot(out[0, active], tone[active]) /
                               np.dot(self.bed[0, active], tone[active]))

        self.assertAlmostEqual(attenuation(self.mid), -2.5*.85, delta=.02)
        self.assertAlmostEqual(attenuation(self.bass), 0., delta=.1)
        self.assertAlmostEqual(attenuation(self.treble), 0., delta=.1)

    def test_single_vocal_reaches_mix_without_added_tracks(self):
        voice = .16*self.mid
        bed = np.zeros_like(self.bed)
        expected, _ = mixing._vocal_chain(voice, self.sr, 75., 1., 1., 1., True)
        out, report = mixing.enhance(voice, bed, self.sr, space=0, gain_db=-2.)
        np.testing.assert_array_equal(out, np.stack([expected]*2)*np.float32(10**(-2/20)))
        self.assertEqual(report['max_low_mid_duck_db'], 0.)
        self.assertEqual(report['max_broadband_duck_db'], 0.)
        self.assertEqual(report['duck_band_hz'], [250, 4000])


if __name__ == '__main__':
    unittest.main()
