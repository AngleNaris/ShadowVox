"""干声分离缓存、处理顺序、空档静音及草稿边界回归。"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shadowvox import config as C, pipeline, separate, vocal_gate
from shadowvox.live_preview import PreviewCache


class DryCleanupTests(unittest.TestCase):
    def test_dry_separation_cache_preserves_float_samples(self):
        vocal = np.linspace(-.15,.15,1000,dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/'recording.wav'
            source.touch()
            (folder/separate.DEFAULT_MODEL).touch()
            with patch.object(C,'MODELS',folder), \
                 patch.object(separate,'separate_song',return_value={'vocals':vocal}) as model:
                first, cached = separate.isolate_dry(source,folder,len(vocal),True,lambda *_: None)
                self.assertFalse(cached)
                second, cached = separate.isolate_dry(source,folder,len(vocal),True,lambda *_: None)
                self.assertTrue(cached)
                model.assert_called_once()
            np.testing.assert_array_equal(first,vocal)
            np.testing.assert_array_equal(second,vocal)

    def test_gap_and_fades_match_when_preview_begins_inside_a_phrase(self):
        sr = 16000
        seconds = np.arange(4*sr)/sr
        singing = ((seconds >= .8) & (seconds < 1.4)) | ((seconds >= 2.5) & (seconds < 3.1))
        f0 = (singing[::160]*220.).astype(np.float32)
        reference = .2*np.sin(2*np.pi*220*seconds)*singing
        spans = vocal_gate.reference_intervals(reference,f0,sr)
        self.assertEqual(len(spans),2)
        voice = np.ones(len(seconds),dtype=np.float32)
        output, _ = vocal_gate.mute_gaps(voice,spans,sr)
        self.assertEqual(output[round(1.9*sr)],0.)
        self.assertEqual(output[round(.8*sr)],1.)
        first = spans[0][0]
        self.assertEqual(output[first],0.)
        self.assertTrue(0. < output[first+100] < 1.)
        local, _ = vocal_gate.mute_gaps(voice[12000:45000],spans,sr,start=12000)
        np.testing.assert_array_equal(local,output[12000:45000])
        shifted, _ = vocal_gate.mute_gaps(voice,spans,sr,start=-3200)
        np.testing.assert_array_equal(shifted[3200:],output[:-3200])

    def test_separation_feeds_denoise_and_final_and_draft_mute_only_vocal(self):
        sr = C.VOC_SR
        t = np.arange(4*sr)/sr
        singing = (t >= 1.) & (t < 2.)
        vocal = (.2*np.sin(2*np.pi*220*t)).astype(np.float32)
        reference = vocal*singing
        bed = np.stack([.1*np.sin(2*np.pi*410*t)]*2).astype(np.float32)
        capture = {}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dry, song, output = folder/'dry.wav',folder/'song.wav',folder/'out.wav'
            sf.write(dry,vocal*.5,sr,subtype='FLOAT')
            sf.write(song,bed.T,sr,subtype='FLOAT')

            class Extractor:
                def extract(self, wave, sample_rate):
                    return np.where(np.abs(wave[::441]).reshape(-1) > 1e-5,220.,0.)

            result = SimpleNamespace(shift_cents=np.zeros(400),confidence=np.zeros(400),
                                     stats={},offset_seconds=0.)
            opts = pipeline.Options(strength=0.,rhythm=False,declip=False,renderer='psola',
                                    dry_has_backing=True,mute_reference_gaps=True)
            with patch.object(pipeline,'isolate_dry',return_value=(vocal,False)), \
                 patch.object(pipeline,'_load_or_separate',return_value=(reference,bed)), \
                 patch.object(pipeline,'FcpeExtractor',Extractor), \
                 patch.object(pipeline,'estimate_shift',return_value=result), \
                 patch('shadowvox.denoise.process',side_effect=lambda voice,sr: voice) as denoise:
                report = pipeline.run(dry,song,output,opts=opts,capture=capture,log=lambda *_: None)
            np.testing.assert_array_equal(denoise.call_args.args[0],vocal)
            self.assertTrue(report['dry_separation']['enabled'])
            self.assertEqual(report['reference_gap_mute']['singing_intervals'],1)
            quiet = slice(3*sr,round(3.5*sr))
            np.testing.assert_array_equal(capture['vocal'][quiet],0.)
            final,_ = sf.read(output,dtype='float32')
            np.testing.assert_allclose(final[quiet],bed.T[quiet],atol=1e-4)
            self.assertGreater(float(np.max(np.abs(capture['vocal'][sr:2*sr]))),.1)
            cache = PreviewCache(capture,report,{'gains_db':[0,0]})
            _, draft = cache.render(opts,2.)
            np.testing.assert_array_equal(draft[quiet],bed.T[quiet])


if __name__ == '__main__':
    unittest.main()
