"""Regression checks for the selected vocal mix and reference-balance curve."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shadowvox import audio, mix_balance, mixing, pipeline
from shadowvox.live_preview import PreviewCache


class SelectedMixTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.arange(12*self.sr) / self.sr
        envelope = np.where((t % 4) < 2.8, .2, .005)
        self.voice = (envelope*np.sin(2*np.pi*220*t)).astype(np.float32)
        self.reference = (.16*np.sin(2*np.pi*220*t)*(envelope > .01)).astype(np.float32)
        self.bed = np.stack([.12*np.sin(2*np.pi*440*t), .1*np.sin(2*np.pi*550*t)]).astype(np.float32)

    def test_reference_curve_is_slow_and_resists_small_timing_offsets(self):
        t, gain, stats = mix_balance.curve(self.voice, np.stack([self.voice]*2), self.bed,
                                           self.reference, self.bed, self.sr)
        shifted = np.r_[np.zeros(self.sr//2,dtype=np.float32), self.voice[:-self.sr//2]]
        _, moved, _ = mix_balance.curve(shifted, np.stack([shifted]*2), self.bed,
                                        self.reference, self.bed, self.sr)
        self.assertTrue(stats['reference_used'])
        self.assertGreater(stats['reference_sections'], 0)
        self.assertLessEqual(float(np.max(np.abs(np.diff(gain)/np.diff(t)))), .20001)
        self.assertLess(float(np.percentile(np.abs(gain-moved),90)), .25)

    def test_manual_level_does_not_get_cancelled_by_reference(self):
        first, second = {}, {}
        _, info = mixing.enhance(self.voice, self.bed, self.sr,
                                  reference_vocal=self.reference, capture=first)
        scaled = self.voice * 10**(6/20)
        mixing.enhance(scaled, self.bed, self.sr, reference_vocal=self.reference,
                       vocal_gain_db=6, capture=second)
        self.assertTrue(info['reference_balance']['reference_used'])
        np.testing.assert_allclose(first['mix_gain_curve']['gain_db'],
                                   second['mix_gain_curve']['gain_db'], atol=1e-4)
        _, missing = mixing.enhance(self.voice, self.bed, self.sr,
                                    reference_vocal=np.zeros_like(self.voice))
        self.assertFalse(missing['reference_balance']['reference_used'])

    def test_selected_short_room_and_wetter_tail_use_summed_vocal(self):
        voice = self.voice[:4*self.sr].copy()
        voice[self.sr:] = 0
        bed = np.zeros((2,len(voice)), dtype=np.float32)
        dry, _ = mixing.enhance(voice, bed, self.sr, space=0, duck=0)
        wet, info = mixing.enhance(voice, bed, self.sr, delay_amount=0, duck=0)
        y = dry[0]
        env = mixing.envelope(y,self.sr,attack=.025,release=.25)
        activity = np.clip((20*np.log10(np.maximum(env,1e-8))+33)/18,0,1)
        wet_gain = 10**(-4.5*activity/20)
        reference = mixing.room(y,self.sr,size=.66)*10**((mixing.LONG_ROOM_RETURN_DB-3)/20)*wet_gain
        result = wet-dry-mixing.short_room(y,self.sr)
        np.testing.assert_allclose(result, reference*10**(3/20), atol=5e-7)
        tail = slice(round(1.4*self.sr),round(1.8*self.sr))
        self.assertGreater(audio.rms(result[:,tail]),1.4*audio.rms(reference[:,tail]))
        self.assertEqual(info['reverb_room_size'],.66)
        self.assertEqual(info['reverb_tail_boost_db'],3.)
        self.assertEqual(info['vocal_body']['preset'],'B_light_third')

    def test_bus_compression_is_stereo_linked_and_master_measures_output(self):
        t = np.arange(4*self.sr)/self.sr
        left = .4*np.sin(2*np.pi*220*t)
        source = np.stack([left,left*.5]).astype(np.float32)
        compressed, stats = mixing.bus_compress(source,self.sr)
        self.assertGreater(stats['max_reduction_db'],0)
        self.assertLessEqual(stats['max_reduction_db'],1.8)
        np.testing.assert_allclose(compressed[1],compressed[0]*.5,atol=1e-7)
        result, report = mixing.master(source,self.sr,apply_glue=True)
        self.assertTrue(report['bus_compression']['enabled'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mix.wav'
            sf.write(path,result.T,self.sr,subtype='PCM_24')
            lufs, peak = mixing.measure_loudness(path)
        self.assertAlmostEqual(lufs,report['estimated_output_lufs'],delta=.15)
        self.assertLessEqual(peak,-1.5)


class PipelineMixTests(unittest.TestCase):
    def test_full_pipeline_passes_reference_and_applies_glue(self):
        sr = 44100
        t = np.arange(6*sr)/sr
        vocal = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        reference = (.15*np.sin(2*np.pi*220*t)).astype(np.float32)
        bed = np.stack([.12*np.sin(2*np.pi*440*t)]*2).astype(np.float32)

        class Extractor:
            def extract(self, signal, sample_rate):
                return np.full(len(signal)//441,220.,dtype=np.float32)

        def alignment(dry, ref, **kwargs):
            self.assertEqual(kwargs['offset_override'], 0.)
            return SimpleNamespace(shift_cents=np.zeros(len(dry)), confidence=np.ones(len(dry)),
                                   stats={}, offset_seconds=0.)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            dry, song, output = folder/'dry.wav', folder/'song.wav', folder/'out.wav'
            sf.write(dry,vocal,sr); sf.write(song,bed.T,sr)
            opts = pipeline.Options(strength=0, denoise=False, declip=False, rhythm=False,
                                    renderer='psola', mix_effects=True)
            capture = {}
            with patch.object(pipeline,'_load_or_separate',return_value=(reference,bed)), \
                 patch.object(pipeline,'FcpeExtractor',Extractor), \
                 patch.object(pipeline,'estimate_shift',side_effect=alignment), \
                 patch('shadowvox.psola.render',side_effect=lambda signal,*args,**kwargs: signal):
                report = pipeline.run(dry,song,output,opts=opts,capture=capture,log=lambda _: None)
            self.assertTrue(report['mixing']['reference_balance']['reference_used'])
            self.assertTrue(report['mixing']['bus_compression']['enabled'])
            self.assertIn('mix_gain_curve',capture)
            self.assertEqual(capture['move'], 0.)
            rendered, output_sr = sf.read(output,always_2d=True)
            self.assertEqual(rendered.shape,(len(vocal),2))
            self.assertEqual(output_sr,sr)
            self.assertTrue(np.isfinite(rendered).all())
            preview = PreviewCache(capture, report, {'gains_db':[0,0]})
            with patch.object(mixing,'enhance',wraps=mixing.enhance) as draft_mix:
                _, first = preview.render(opts,0,seconds=4)
            self.assertEqual(draft_mix.call_args.kwargs['body_gains'],
                             report['mixing']['vocal_body']['normalization'])
            capture['mix_gain_curve']['gain_db'] -= 1
            _, softer = preview.render(opts,0,seconds=4)
            self.assertEqual(first.shape,softer.shape)
            self.assertGreater(float(np.max(np.abs(first-softer))),1e-5)
            capture.pop('mix_gain_curve')
            _, fallback = preview.render(opts,0,seconds=4)
            self.assertTrue(np.isfinite(fallback).all())
            with patch.object(pipeline,'_load_or_separate',return_value=(reference,bed)), \
                 patch.object(pipeline,'FcpeExtractor',Extractor), \
                 patch.object(pipeline,'estimate_shift',side_effect=alignment), \
                 patch('shadowvox.psola.render',side_effect=lambda signal,*args,**kwargs: signal):
                bypassed = pipeline.run(dry,song,folder/'bypass.wav',
                                        opts=replace(opts,mix_effects=False),log=lambda _: None)
            self.assertFalse(bypassed['mixing']['enabled'])


if __name__ == '__main__':
    unittest.main()
