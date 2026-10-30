"""已选 B 增厚的频率/相位、整曲标定与统一路由回归。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import parselmouth
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shadowvox import audio, mixing, vocal_body


class VocalBodyTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.t = np.arange(2*self.sr)/self.sr
        self.dry = (.15*np.cos(2*np.pi*250*self.t+.7)).astype(np.float32)

    def test_second_and_third_keep_their_known_source_phases(self):
        sound = parselmouth.Sound(self.dry.astype(np.float64),sampling_frequency=self.sr)
        phase, region, _ = vocal_body._source_cycles(sound,len(self.dry),self.sr)
        sub = (.04*np.cos(2*np.pi*125*self.t+.2)).astype(np.float32)
        second = vocal_body._second_partial(self.dry,sub,phase,region,np.ones_like(sub),self.sr)
        third = vocal_body._third_partial(sub,phase,region,self.sr)
        sl = slice(round(.2*self.sr),round(1.8*self.sr))
        self.assertGreater(np.corrcoef(second[sl],self.dry[sl])[0,1],.99)
        expected = np.cos(2*np.pi*375*self.t+.6)
        self.assertGreater(np.corrcoef(third[sl],expected[sl])[0,1],.99)
        frequencies, power = welch(third[sl],self.sr,nperseg=self.sr)
        self.assertAlmostEqual(frequencies[np.argmax(power)],375.,delta=1.)
        self.assertLess(power[frequencies >= 800].sum()/power.sum(),1e-4)

    def test_octave_quantity_and_fixed_normalization_survive_level_change(self):
        summed, stats = vocal_body.process(self.dry,self.sr)
        self.assertTrue(stats['enabled'])
        delta = summed-self.dry*10**(vocal_body.SOURCE_TRIM_DB/20)
        sl = slice(round(.2*self.sr),round(1.8*self.sr))
        frequencies, power = welch(delta[sl],self.sr,nperseg=self.sr)
        self.assertEqual(frequencies[np.argmax(power)],125.)
        sub = (frequencies >= 100) & (frequencies <= 150)
        total = power[sub].sum()
        # 原声 .15 振幅，已选低八度约 -11.84 dB（含辅轨补偿与共同输入修整）。
        expected = .15**2/2*10**((-12+.35-.19)/10)
        self.assertAlmostEqual(10*np.log10(total/expected),0.,delta=.3)
        reused, reused_stats = vocal_body.process(self.dry*.5,self.sr,
                                                  normalization=stats['normalization'])
        self.assertTrue(reused_stats['normalization_reused'])
        self.assertEqual(reused_stats['normalization'],stats['normalization'])
        np.testing.assert_allclose(reused[sl],summed[sl]*.5,atol=5e-6)

    def test_silence_noise_and_short_input_do_not_create_a_low_voice(self):
        noise = np.random.default_rng(16).normal(0,.01,len(self.dry)).astype(np.float32)
        for source in (np.zeros_like(self.dry),noise,self.dry[:100]):
            result, stats = vocal_body.process(source,self.sr)
            self.assertFalse(stats['enabled'])
            np.testing.assert_array_equal(result,source)

    def test_all_space_and_backing_sidechains_receive_the_summed_voice(self):
        summed = (self.dry+.06*np.sin(2*np.pi*125*self.t)).astype(np.float32)
        bed = np.zeros((2,len(summed)),dtype=np.float32)
        with patch.object(vocal_body,'process',return_value=(summed,{})), \
             patch.object(mixing,'short_room',return_value=bed) as short, \
             patch.object(mixing,'room',return_value=bed) as long, \
             patch.object(mixing,'delay',return_value=bed) as delay, \
             patch.object(mixing,'_backing',return_value=(bed,np.zeros(len(summed)))) as backing:
            result, _ = mixing.enhance(self.dry,bed,self.sr,gain_db=-2.)
        for effect in (short,long,delay):
            np.testing.assert_array_equal(effect.call_args.args[0],summed)
        np.testing.assert_array_equal(backing.call_args.args[1],summed)
        np.testing.assert_allclose(result,np.stack([summed]*2)*10**(-2/20),atol=1e-7)


if __name__ == '__main__':
    unittest.main()
