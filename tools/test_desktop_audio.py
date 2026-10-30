"""有限节奏与轻度混音的可验证行为。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from shadowvox.guarded_rhythm import plan, align
from shadowvox.mixing import enhance, compress, room


def melody(delays=None, missing=None):
    f = np.zeros(1500,dtype=np.float32)
    for i,(start,hz) in enumerate(zip([150,350,550,750,950,1150],[180,220,260,200,240,300])):
        if i == missing:
            continue
        a=start+(delays or {}).get(i,0)
        f[a:a+60]=hz*(1+.01*np.sin(np.arange(60)/8))
    return f


class GuardTests(unittest.TestCase):
    def test_reliable_global_delay_and_sparse_abstention(self):
        ref=melody();dry=melody(dict.fromkeys(range(6),35))
        _,_,report=plan(dry,ref)
        self.assertTrue(report['global_offset_reliable'])
        self.assertAlmostEqual(report['placement_offset_seconds'],.35,places=2)
        ref=np.zeros(1000);dry=np.zeros(1000)
        ref[100:800]=220;dry[105:805]=220
        lead,mask,report=plan(dry,ref)
        self.assertFalse(report['global_offset_reliable'])
        self.assertFalse(report['applied'])
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(lead,0)

    def test_small_offsets_and_bound(self):
        ref=melody();dry=melody({1:8,3:-6})
        lead,mask,report=plan(dry,ref,offset_override=0)
        self.assertGreater(lead[358],0)
        self.assertLess(lead[744],0)
        self.assertLessEqual(np.max(np.abs(lead)),.15)
        self.assertLessEqual(np.max(np.abs(np.diff(lead))),.001001)
        self.assertGreater(report['onset_pairs'],3)

    def test_large_offset_skipped_and_next_phrase_independent(self):
        ref=melody();dry=melody({2:60,3:5})
        lead,mask,report=plan(dry,ref,offset_override=0)
        np.testing.assert_array_equal(lead[610:670],0)
        self.assertFalse(mask[610:670].any())
        self.assertGreater(lead[755],0)
        self.assertGreater(report['skipped_onsets'],0)

    def test_missing_phrase_stays_empty(self):
        ref=melody();dry=melody({3:5},missing=2)
        lead,mask,_=plan(dry,ref,offset_override=0)
        np.testing.assert_array_equal(lead[550:610],0)
        self.assertFalse(mask[550:610].any())
        self.assertGreater(lead[755],0)

    def test_wrong_melody_and_no_voice(self):
        ref=melody();dry=melody();dry[350:410]*=2**(5/12)
        lead,mask,_=plan(dry,ref,offset_override=0)
        self.assertFalse(mask[350:410].any())
        lead,_,report=plan(np.zeros(1500),ref)
        self.assertFalse(report['applied'])
        self.assertEqual(report['placement_offset_seconds'],0)

    def test_waveform_protected_and_length(self):
        sr=8000;ref=melody();dry=melody({1:8,2:60,3:-6})
        t=np.arange(15*sr)/sr;x=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
        y,stats=align(x,dry,ref,sr,log=lambda _:None)
        self.assertEqual(len(y),len(x));self.assertTrue(np.isfinite(y).all())
        for a,b in stats['protected_intervals_seconds']:
            np.testing.assert_array_equal(y[round(a*sr):round(b*sr)],x[round(a*sr):round(b*sr)])


class MixingTests(unittest.TestCase):
    def test_silence_leaves_accompaniment_and_length(self):
        sr=16000;t=np.arange(sr*2)/sr;bed=np.array([.1*np.sin(2*np.pi*500*t)]*2,dtype=np.float32)
        out,stats=enhance(np.zeros(len(t),dtype=np.float32),bed,sr)
        np.testing.assert_allclose(out,bed,atol=1e-6)
        self.assertEqual(stats['max_band_duck_db'],0)

    def test_compression_reverb_and_duck(self):
        sr=16000;t=np.arange(sr*2)/sr
        voice=(.5*np.sin(2*np.pi*220*t)).astype(np.float32);voice[sr:]=0
        compressed,reduction=compress(voice,sr)
        self.assertGreater(reduction,3);self.assertLess(reduction,6.01)
        self.assertLess(np.std(compressed[:sr]),np.std(voice[:sr]))
        wet=room(voice,sr)
        self.assertGreater(np.max(np.abs(wet[:,sr:sr+4000])),.0001)
        out,stats=enhance(voice,np.zeros((2,len(t)),np.float32),sr)
        self.assertEqual(out.shape,(2,len(t)));self.assertTrue(np.isfinite(out).all())
        self.assertGreater(stats['max_band_duck_db'],2)


if __name__=='__main__':
    unittest.main()
