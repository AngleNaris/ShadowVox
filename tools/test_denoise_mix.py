"""验证降噪顺序、时间轴、混音峰值和插件状态保存。"""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from shadowvox import denoise, mixing, pipeline, repair
from shadowvox.vst_host import open_editor


class DenoiseTests(unittest.TestCase):
    def test_length_and_alignment_at_project_sample_rate(self):
        sr = 44100
        t = np.arange(sr*2+137)/sr
        # 变化的基频和包络使互相关不容易匹配到相邻周期。
        voice = (.2*np.sin(2*np.pi*(180*t+18*t*t))*(.5+.5*np.sin(2*np.pi*.7*t))).astype(np.float32)
        result = denoise.process(voice,sr)
        self.assertEqual(len(result),len(voice))
        self.assertTrue(np.isfinite(result).all())
        source_8k = resample_poly(voice,80,441)
        output_8k = resample_poly(result,80,441)
        correlation = correlate(output_8k,source_8k,mode="full",method="fft")
        lag = correlation_lags(len(output_8k),len(source_8k))[np.argmax(correlation)]
        self.assertLessEqual(abs(lag),1)
        self.assertGreater(np.max(np.abs(result[-sr//10:])),1e-4)

    def test_denoise_runs_before_pitch_extraction_and_can_be_bypassed(self):
        events = []
        original = np.ones(44100,dtype=np.float32)

        def clean(samples, sr):
            events.append("denoise")
            return samples*.5

        class Extractor:
            def extract(self, samples, sr):
                events.append(("pitch",float(samples[0])))
                raise RuntimeError("stop after first pitch")

        for enabled in (True,False):
            events.clear()
            with patch.object(pipeline.audio,"load_mono",return_value=original), \
                 patch.object(pipeline,"_load_or_separate",return_value=(original,np.zeros((2,len(original))))), \
                 patch.object(repair,"repair_clipping",return_value=(original,{"applied":False})), \
                 patch.object(denoise,"process",side_effect=clean), \
                 patch.object(pipeline,"FcpeExtractor",Extractor):
                with self.assertRaisesRegex(RuntimeError,"stop after first pitch"):
                    pipeline.run("dry.wav","song.wav","result.wav",opts=pipeline.Options(denoise=enabled),log=lambda _:None)
            self.assertEqual(events, ["denoise",("pitch",.5)] if enabled else [("pitch",1.)])

    def test_declip_only_on_obvious_clipping(self):
        sr = 44100
        t = np.arange(sr*3)/sr
        source = (.45*np.sin(2*np.pi*(180*t+2*t*t))+.16*np.sin(2*np.pi*830*t)).astype(np.float32)
        unchanged, stats = repair.repair_clipping(source,sr)
        self.assertFalse(stats['applied'])
        np.testing.assert_array_equal(unchanged,source)
        normalized_sine = np.sin(2*np.pi*100*t).astype(np.float32)
        unchanged, stats = repair.repair_clipping(normalized_sine,sr)
        self.assertFalse(stats['applied'])
        np.testing.assert_array_equal(unchanged,normalized_sine)
        reference = source/.30
        clipped = np.clip(reference,-1,1)
        restored, stats = repair.repair_clipping(clipped,sr)
        self.assertTrue(stats['applied'])
        self.assertEqual(len(restored),len(source))
        self.assertLessEqual(np.max(np.abs(restored)),1.)
        restored *= np.sum(reference*restored)/np.sum(restored*restored)
        self.assertLess(np.mean((restored-reference)**2),np.mean((clipped-reference)**2))


class MixTests(unittest.TestCase):
    def test_single_gentle_deess_after_all_eq(self):
        sr = 44100
        t = np.arange(sr) / sr
        voice = (.2 * np.sin(2*np.pi*250*t)).astype(np.float32)
        voice[sr//3:2*sr//3] += (.3*np.sin(2*np.pi*7000*t[sr//3:2*sr//3])).astype(np.float32)
        events = []
        original_bell, original_deess = mixing.bell, mixing.deess
        original_low, original_high = mixing.LowShelfFilter, mixing.HighShelfFilter

        def bell(*args, **kwargs):
            events.append("bell")
            return original_bell(*args, **kwargs)

        def low(*args, **kwargs):
            events.append("low_shelf")
            return original_low(*args, **kwargs)

        def high(*args, **kwargs):
            events.append("high_shelf")
            return original_high(*args, **kwargs)

        def deess(*args, **kwargs):
            events.append("deess")
            return original_deess(*args, **kwargs)

        with patch.object(mixing,"bell",side_effect=bell), \
             patch.object(mixing,"LowShelfFilter",side_effect=low), \
             patch.object(mixing,"HighShelfFilter",side_effect=high), \
             patch.object(mixing,"deess",side_effect=deess):
            result, info = mixing.enhance(voice,np.zeros((2,len(voice)),np.float32),sr,space=0)
        self.assertEqual(events.count("deess"),1)
        self.assertGreater(events.index("deess"),max(i for i,x in enumerate(events) if x != "deess"))
        self.assertEqual(info["deess_position"],"after_eq")
        self.assertEqual(info["deess_strength"],.3)
        self.assertLessEqual(info["max_deess_db"],1.02)
        self.assertEqual(result.shape,(2,len(voice)))
        self.assertTrue(np.isfinite(result).all())

    def test_reverb_tail_and_space_control(self):
        sr = 44100
        t = np.arange(sr * 3) / sr
        voice = (.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        voice[sr:] = 0
        wet = mixing.room(voice, sr)
        self.assertEqual(wet.shape, (2, len(voice)))
        self.assertEqual(np.max(np.abs(wet[:, :round(.02 * sr)])), 0)
        self.assertGreater(np.max(np.abs(wet[:, sr + sr//2:sr + 3*sr//4])), 1e-4)
        self.assertGreater(np.max(np.abs(wet[0] - wet[1])), 1e-4)

        bed = np.zeros((2, len(voice)), dtype=np.float32)
        dry, _ = mixing.enhance(voice, bed, sr, space=0)
        half, _ = mixing.enhance(voice, bed, sr, space=.5)
        normal, info = mixing.enhance(voice, bed, sr, space=1)
        twice, _ = mixing.enhance(voice, bed, sr, space=2)
        np.testing.assert_allclose((half-dry)*2, normal-dry, atol=3e-7)
        np.testing.assert_allclose((normal-dry)*2, twice-dry, atol=3e-7)
        self.assertEqual(info['reverb_engine'], 'Freeverb')

    def test_deess_and_separate_stereo_delay(self):
        sr = 44100
        t = np.arange(sr*2)/sr
        low = (.2*np.sin(2*np.pi*250*t)).astype(np.float32)
        high = (.3*np.sin(2*np.pi*7000*t)).astype(np.float32)
        clean, reduction = mixing.deess(low+high,sr)
        self.assertGreater(reduction,1.)
        self.assertLess(np.std(clean-low),np.std(high))
        only_low, _ = mixing.deess(low,sr)
        self.assertLess(np.max(np.abs(only_low-low)),.003)
        voice = low.copy(); voice[sr:] = 0
        wet = mixing.delay(voice,sr)
        self.assertGreater(np.max(np.abs(wet[:,sr:sr+sr//5])),1e-5)
        self.assertFalse(np.array_equal(wet[0],wet[1]))
        without, no_space = mixing.enhance(voice,np.zeros((2,len(t)),np.float32),sr,space=0.)
        with_space, normal_space = mixing.enhance(voice,np.zeros((2,len(t)),np.float32),sr,space=1.)
        self.assertEqual(no_space['reverb_wet'],0.)
        self.assertGreater(normal_space['reverb_wet'],0.)
        self.assertGreater(np.max(np.abs(with_space-without)),1e-4)

    def test_master_true_peak_and_delay(self):
        sr = 44100
        t = np.arange(sr*4)/sr
        source = np.array([.01*np.sin(2*np.pi*333*t)]*2,dtype=np.float32)
        source[:,sr] = .8
        result, stats = mixing.master(source,sr)
        self.assertTrue(stats['limiter_used'])
        self.assertEqual(result.shape,source.shape)
        self.assertEqual(np.argmax(np.abs(source[0])),np.argmax(np.abs(result[0])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"master.wav"
            sf.write(path,result.T,sr,subtype="PCM_16")
            _, true_peak = mixing.measure_loudness(path)
        self.assertLessEqual(true_peak,-1.48)


class PluginTests(unittest.TestCase):
    def test_state_round_trip_without_audio_processing(self):
        class Parameter:
            is_automatable = True
            def __init__(self): self.raw_value = .8

        class Plugin:
            def __init__(self, expected, new):
                self.raw_state = b"opaque-state"
                self.parameters = {"threshold":Parameter()}
                self.expected, self.new = expected, new
            def show_editor(self, close_event):
                assert self.parameters["threshold"].raw_value==self.expected
                self.parameters["threshold"].raw_value=self.new

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"example.vst3"
            path.touch()
            state = Path(directory)/"plugin.state"
            with patch("pedalboard.load_plugin",side_effect=[Plugin(.8,.4),Plugin(.4,.2)]):
                open_editor(path,state)
                self.assertEqual(state.read_bytes(),b"opaque-state")
                open_editor(path,state)
                saved=json.loads(state.with_suffix('.params.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['threshold'],.2)


if __name__ == "__main__":
    unittest.main()
