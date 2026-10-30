"""中文音素对应与连续变时的已知答案验证。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shadowvox import hubert_alignment as hfa
from shadowvox.full_align import pitch_mapping, render_segments
from shadowvox.live_preview import PreviewCache
from shadowvox.pipeline import Options


def word(text,t):
    return dict(text=text,start=t,end=t+.3,final_start=t+.08,final_end=t+.3)


class HubertTests(unittest.TestCase):
    def test_missing_word_does_not_shift_following_word_identity(self):
        left=[word(s,i*.4) for i,s in enumerate(['wo','ai','ni','jin','tian','hen','hao'])]
        right=[dict(w,start=w['start']+.06,end=w['end']+.06,final_start=w['final_start']+.06,
                    final_end=w['final_end']+.06) for w in left]
        right.insert(3,word('bu',1.12))
        groups=hfa.word_groups(left,right)
        self.assertEqual([len(g) for g in groups],[3,4])
        self.assertTrue(all(a['text']==b['text'] for g in groups for a,b in g))

    def test_repeated_single_syllable_or_far_repeat_is_not_evidence(self):
        words=[word('la',i*.4) for i in range(8)]
        self.assertEqual(hfa.word_groups(words,words),[])
        words=[word(s,i*.4) for i,s in enumerate(['wo','ai','ni'])]
        other=[word(w['text'],w['start']+4) for w in words]
        self.assertEqual(hfa.word_groups(words,other),[])

    def test_abnormal_consonant_uses_final_and_short_final_is_rejected(self):
        a,b=word('mei',1.),word('mei',.75)
        b.update(final_start=1.12,final_end=1.35,end=1.35)
        anchors,trusted=hfa.anchors_for_words([(a,b)])
        self.assertEqual(anchors,[(1.08,1.12)])
        b['final_end']=b['final_start']+.005
        self.assertEqual(hfa.anchors_for_words([(a,b)]),([],[]))

    def test_fit_obeys_limits_and_strength_without_reordering(self):
        source=np.array([0.,.3,.6,.8,1.2,1.8,2.])
        reference=source+np.array([0.,.3,-.2,.2,-.2,.1,0.])
        full=hfa.fit(source,reference)
        self.assertLessEqual(max(abs(full-source)),.15000001)
        rates=np.diff(full)/np.diff(source)
        self.assertGreaterEqual(min(rates),.9-1e-7)
        self.assertLessEqual(max(rates),1.1+1e-7)
        np.testing.assert_allclose(hfa.fit(source,reference,.5)-source,(full-source)*.5,atol=1e-12)
        np.testing.assert_array_equal(hfa.fit(source,reference,0),source)
        np.testing.assert_array_equal(full[[0,-1]],source[[0,-1]])

    def test_pitch_correspondence_survives_zero_rhythm_strength(self):
        source=np.array([0.,.5,1.,1.5,2.]);reference=source+np.array([0.,.1,.1,.1,0.])
        plan=dict(windows=[dict(source=source,reference=reference)],dry_seconds=2.,ref_seconds=2.,
                  dry_t=source,ref_t=reference,offset=0.,acoustic_trusted=np.ones(20,bool))
        pieces,stats=hfa.plan_segments(plan,strength=0)
        times,valid=pitch_mapping(pieces,plan,200)
        self.assertAlmostEqual(times[100],1.1)
        self.assertTrue(valid[100]);self.assertFalse(stats['applied'])

    def test_continuous_warp_renders_once_and_preview_uses_same_waveform(self):
        x=np.arange(200,dtype=np.float32)
        pieces=[dict(source=np.array([0.,1.,2.]),target=np.array([0.,1.05,2.]),
                     protected=False,continuous=True)]
        with patch('shadowvox.full_align.stretch',return_value=x+1) as renderer:
            full,_=render_segments(x,pieces,100)
            partial,_=render_segments(x,pieces,100,start=47,end=131)
        np.testing.assert_array_equal(partial,full[47:131])
        self.assertEqual(renderer.call_count,2)
        self.assertEqual(len(full),len(x))

    def test_preview_dispatches_to_same_chinese_plan(self):
        plan=dict(backend='hubertfa_zh')
        cache=PreviewCache(dict(options={'declip':False},rhythm_plan=plan),{}, {'gains_db':[0,0]})
        with patch('shadowvox.hubert_alignment.plan_segments',return_value=(['piece'],{'test':True})) as fn:
            pieces,stats,offset=cache.plan(Options(rhythm_strength=.5),np.zeros(20))
        self.assertEqual(pieces,['piece']);self.assertEqual(offset,0)
        self.assertEqual(fn.call_args.args,(plan,.5,.1))

    def test_recognizer_is_explicitly_chinese_and_filters_hallucination(self):
        model=SimpleNamespace(transcribe=lambda *args,**kw: self.transcribe(kw))
        self.assertEqual(hfa._text(model,np.zeros(16000),16000),'爱你')

    def transcribe(self,kw):
        self.assertEqual(kw['language'],'zh')
        return {'segments':[{'text':'爱你','avg_logprob':-.6,'no_speech_prob':.8},
                            {'text':'字幕','avg_logprob':-1.5}]}


if __name__=='__main__':unittest.main()
