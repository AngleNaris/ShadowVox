"""结果报告的已知答案标定：偏移、实际音高、共同落点和纯增益不变性。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shadowvox import report_analysis as analysis
from shadowvox.reporting import build, html_report, visual_payload


class ResultsTests(unittest.TestCase):
    def test_actual_pitch_comparison_uses_common_frames_and_preserves_octave(self):
        reference = np.full(1200,110.)
        before = np.full(1200,220.*2**(35/1200))
        after = np.full(1200,220.*2**(5/1200))
        after[:80] = 0
        result = analysis.pitch_result(before,after,reference)
        self.assertEqual(result['common_frames'],1120)
        self.assertAlmostEqual(result['before_median'],35.)
        self.assertAlmostEqual(result['after_median'],5.)
        self.assertGreater(min(v for v in result['window']['reference'] if v),56.)
        self.assertFalse(analysis.pitch_result(np.zeros(50),np.zeros(50),np.zeros(50))['available'])

    def test_placement_sign_and_common_actual_onsets_are_calibrated(self):
        starts = [100,220,340,450,580,710,860,990,1120,1260]
        before_offsets = [8,-6,10,-9,7,-8,9,-7,6,-10]
        after_offsets = [1,-1,0,-1,1,0,1,-1,0,1]
        signals = []
        for offsets in ([0]*10,before_offsets,after_offsets):
            f0 = np.zeros(1400)
            for start,offset in zip(starts,offsets):
                f0[start+offset:start+offset+30] = 220
            signals.append(f0)
        reference,before,after = signals
        result = analysis.timing_result(before,after,reference)
        self.assertTrue(result['available'])
        self.assertEqual(result['paired_onsets'],10)
        np.testing.assert_allclose([p['before'] for p in result['pairs']],np.array(before_offsets)*10)
        np.testing.assert_allclose([p['after'] for p in result['pairs']],np.array(after_offsets)*10)
        late = np.r_[np.zeros(8),reference[:-8]]
        np.testing.assert_array_equal(analysis._placed_frames(late,-.08,len(reference)),reference)
        unchanged = analysis.timing_result(before,before,reference)
        self.assertEqual(unchanged['before_median_ms'],unchanged['after_median_ms'])

    def test_vocal_tone_does_not_claim_a_gain_change_as_timbre(self):
        sr = 44100
        t = np.arange(2*sr)/sr
        source = (.10*np.sin(2*np.pi*250*t)+.06*np.sin(2*np.pi*3000*t)+.03*np.sin(2*np.pi*9000*t)).astype(np.float32)
        a = analysis.voice_features(source,sr)
        b = analysis.voice_features(source*.5,sr)
        np.testing.assert_allclose(a['spectrum'],b['spectrum'],atol=.02)
        self.assertEqual(a['dynamic_range_db'],b['dynamic_range_db'])
        changed = source+.1*np.sin(2*np.pi*250*t)
        c = analysis.voice_features(changed,sr)
        self.assertGreater(c['bands']['声体']-a['bands']['声体'],1.)
        self.assertLess(c['bands']['空气感']-a['bands']['空气感'],-1.)

    def test_stereo_field_uses_power_and_is_level_invariant(self):
        sr = 16000
        t = np.arange(2*sr)/sr
        left = .15*np.sin(2*np.pi*423*t)
        right = .12*np.sin(2*np.pi*551*t)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f'{n}.wav' for n in range(4)]
            for path,data in zip(paths,(np.stack([left,right],axis=1),np.stack([left,right],axis=1)*.5,
                                         np.stack([left,left],axis=1),np.stack([left,-left],axis=1))):
                sf.write(path,data,sr,subtype='FLOAT')
            a,b,mono,inverse = (analysis.stereo_scope(p) for p in paths)
        np.testing.assert_allclose(a['points'],b['points'],atol=.001)
        self.assertEqual(a['side_energy_percent'],b['side_energy_percent'])
        self.assertEqual(mono['side_energy_percent'],0.)
        self.assertEqual(inverse['side_energy_percent'],100.)

    def test_missing_analysis_and_offline_html_remain_truthful(self):
        sr = 16000
        t = np.arange(2*sr)/sr
        signal = .1*np.sin(2*np.pi*440*t)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.wav'; sf.write(path,signal,sr)
            report = {'song':'安全<script>测试.wav','options':{'strength':0,'rhythm':False}}
            metrics = build(report,path,path)
            metrics['visual']['art'] = 'data:image/png;base64,retired'
            self.assertNotIn('art',visual_payload(report,metrics,1,path)['visual'])
            page = html_report(report,metrics,'<output>.wav')
        self.assertFalse(metrics['visual']['pitch']['available'])
        self.assertFalse(metrics['visual']['timing']['available'])
        self.assertFalse(metrics['visual']['tone']['available'])
        json.dumps(metrics,allow_nan=False)
        self.assertIn('&lt;output&gt;.wav',page)
        self.assertNotIn('安全<script>',page)
        self.assertIn('ShadowReport.png',page)
        self.assertNotIn('data:image/png;base64,retired',page)


if __name__ == '__main__':
    unittest.main()
