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


def phrase_fixture(offsets, second_start=3.4, second_delay=0.):
    first = [word(text,1.+i*.32) for i,text in enumerate(('wo','ai','ni'))]
    second = [word(text,second_start+i*.32) for i,text in enumerate(('jin','tian','hao'))]
    sung = [word(w['text'],w['start']+delay) for w,delay in zip(first,offsets)]
    late = [word(w['text'],w['start']+second_delay) for w in second]
    left,right = sung+late,first+second

    def activity(words):
        active = np.zeros(60,bool)
        for w in words:
            lo = max(0,int(np.floor(w['start']*10))-1)
            hi = min(len(active),int(np.ceil(w['end']*10))+1)
            active[lo:hi] = True
        return active

    groups = [(0.,6.,g) for g in hfa.word_groups(left,right)]
    return hfa.build_plan(groups,activity(left),activity(right),6.,6.)


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

    def test_distinct_consistent_two_syllables_match_without_nearby_alternative(self):
        left = [word('wo',1.),word('ai',1.25)]
        right = [word('wo',1.2),word('ai',1.45)]
        self.assertEqual([len(g) for g in hfa.word_groups(left,right)],[2])
        competing = [word('wo',.8),word('ai',1.05)]+right
        self.assertEqual(hfa.word_groups(left,competing),[])
        self.assertEqual(hfa.word_groups(left,[word('wo',1.2),word('ai',1.25)]),[])

    def test_first_phrase_has_independent_early_and_late_moves(self):
        for delay in (-.4,-.2,-.1,.1,.2,.4):
            with self.subTest(delay=delay):
                plan = phrase_fixture([delay]*3)
                pieces,stats = hfa.plan_segments(plan)
                self.assertAlmostEqual(stats['global_offset_seconds'],0.)
                first = stats['first_phrase']
                self.assertEqual(first['status'],'moved')
                self.assertAlmostEqual(first['residual_before_ms'],delay*1000)
                self.assertAlmostEqual(first['phrase_move_ms'],-delay*1000)
                self.assertAlmostEqual(first['residual_after_ms'],0.)
                self.assertEqual(len(stats['phrase_moves']),1)
                self.assertAlmostEqual(stats['rate_min'],1.)
                self.assertAlmostEqual(stats['rate_max'],1.)
                for previous,following in zip(pieces,pieces[1:]):
                    self.assertAlmostEqual(previous['source'][-1],following['source'][0])
                    self.assertAlmostEqual(previous['target'][-1],following['target'][0])
                    self.assertTrue(np.all(np.diff(previous['target']) > 0))
                self.assertAlmostEqual(pieces[-1]['target'][-1],6.)
                _,half = hfa.plan_segments(plan,strength=.5)
                self.assertAlmostEqual(half['first_phrase']['residual_after_ms'],delay*500)

    def test_real_samples_are_moved_without_stretch_and_partial_matches_full(self):
        sr = 8000
        t = np.arange(round(.3*sr))/sr
        syllables = [(np.sin(2*np.pi*f*t)*np.hanning(len(t))).astype(np.float32)
                     for f in (220,277,330,262,311,392)]

        def track(delay):
            x = np.zeros(6*sr,np.float32)
            starts = [1.+delay+i*.32 for i in range(3)]+[3.4+i*.32 for i in range(3)]
            for begin,syllable in zip(starts,syllables):
                a = round(begin*sr)
                x[a:a+len(syllable)] = syllable
            return x

        expected = track(0.)
        for delay in (-.4,-.2,-.1,.1,.2,.4):
            with self.subTest(delay=delay), patch('shadowvox.full_align.stretch') as stretch:
                pieces,_ = hfa.plan_segments(phrase_fixture([delay]*3))
                full,_ = render_segments(track(delay),pieces,sr)
                a,b = round(.78*sr),round(3.7*sr)
                partial,_ = render_segments(track(delay),pieces,sr,start=a,end=b)
                stretch.assert_not_called()
                np.testing.assert_array_equal(full,expected)
                np.testing.assert_array_equal(partial,full[a:b])

    def test_only_full_voice_phrase_is_stretched_and_bad_length_keeps_translation(self):
        sr = 8000
        dry = np.ones(6*sr,dtype=np.float32)
        plan = phrase_fixture([.4,.42,.38])
        pieces,_ = hfa.plan_segments(plan)
        lo,hi = plan['phrases'][0]['source_interval']
        with patch('shadowvox.full_align.stretch',side_effect=lambda x,*_,**__: x[:-12]) as stretch:
            full,_ = render_segments(dry,pieces,sr)
            a,b = round(.8*sr),round(2.8*sr)
            partial,_ = render_segments(dry,hfa.plan_segments(plan)[0],sr,start=a,end=b)
        self.assertEqual(stretch.call_count,2)
        self.assertEqual(len(stretch.call_args.args[0]),round(hi*sr)-round(lo*sr))
        self.assertTrue(any(p.get('render_fallback') == 'phrase_render_length' for p in pieces))
        self.assertEqual(len(full),len(dry))
        np.testing.assert_array_equal(partial,full[a:b])

    def test_only_first_word_late_does_not_move_the_phrase(self):
        plan = phrase_fixture([.2,0.,0.])
        _,stats = hfa.plan_segments(plan)
        first = stats['first_phrase']
        self.assertEqual(first['status'],'inconsistent_words')
        self.assertEqual(first['phrase_move_ms'],0.)
        self.assertGreater(first['residual_after_ms'],120.)
        self.assertTrue(stats['vocal_preserve_intervals_seconds'])
        self.assertEqual(stats['rendering'],'continuous')
        self.assertGreater(stats['max_intra_phrase_shift_ms'],0.)
        self.assertLessEqual(stats['max_intra_phrase_shift_ms'],150.)

    def test_unrecognized_entry_needs_lyrics_and_matching_activity_boundaries(self):
        times = [2.4,2.8,3.24,3.46,3.87,4.25]
        words = [(word(text,t),word(text,t-.2)) for text,t in zip(('mei','you','gao','su','ni','hao'),times)]
        source = (1.4,4.8)
        supported = hfa._phrase_evidence(source,words,[(1.2,4.6)],0.)
        self.assertEqual(supported['status'],'candidate')
        self.assertEqual(supported['onset_basis'],'activity_supported_by_lyrics')
        self.assertAlmostEqual(supported['requested_move'],-.2)
        uncertain = hfa._phrase_evidence(source,words,[(1.4,4.6)],0.)
        self.assertEqual(uncertain['status'],'uncertain_boundaries')
        self.assertIsNone(uncertain['first'])

    def test_pause_reserve_prevents_borrowing_voiced_time(self):
        plan = phrase_fixture([-.4]*3,second_start=2.58,second_delay=-.45)
        _,stats = hfa.plan_segments(plan)
        self.assertEqual(stats['first_phrase']['status'],'insufficient_pause')
        self.assertEqual(stats['phrase_moves'],[])
        self.assertTrue(stats['vocal_preserve_intervals_seconds'])
        self.assertGreaterEqual(stats['rate_min'],.9)
        self.assertLessEqual(stats['rate_max'],1.1)

    def test_unmatched_activity_is_diagnostic_without_gate_override(self):
        plan = hfa.build_plan([],np.r_[np.zeros(10,bool),np.ones(10,bool)],np.ones(20,bool),2.,2.)
        _,stats = hfa.plan_segments(plan)
        self.assertIsNone(stats['first_phrase']['residual_before_ms'])
        self.assertEqual(stats['unmatched_intervals_seconds'],[[1.,2.]])
        self.assertEqual(stats['vocal_preserve_intervals_seconds'],[])

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

    def test_consistent_phrases_allow_a_large_global_shift(self):
        first = [word(text,1.+i*.3) for i,text in enumerate(('wo','ai','ni'))]
        second = [word(text,4.+i*.3) for i,text in enumerate(('jin','tian','hao'))]
        groups = []
        for phrase in (first,second):
            delayed = [word(w['text'],w['start']+.65) for w in phrase]
            self.assertEqual(hfa.word_groups(delayed,phrase),[])
            matched, = hfa.word_groups(delayed,phrase,-.65)
            groups.append((0.,8.,matched))
        self.assertEqual(hfa._confirmed_offset(groups[:1],-.65),0.)
        self.assertAlmostEqual(hfa._confirmed_offset(groups,-.65),-.65)
        self.assertEqual(hfa._confirmed_offset(groups,-.65,[(0.,8.)]),0.)

    def test_sub_grid_vowel_and_placement_have_distinct_coordinates(self):
        source = np.array([0.,2.])
        for offset in (-.6,.6):
            pairs = np.array([[1.03,1.08,1.03+offset,1.10+offset]])
            plan = dict(windows=[],dry_seconds=2.,ref_seconds=3.,offset=offset,pitch_pairs=pairs)
            pieces,stats = hfa.plan_segments(plan,.5)
            placement = max(0.,offset*.5)
            self.assertAlmostEqual(stats['median_offset_seconds'],offset*.5)
            x = np.arange(200,dtype=np.float32)
            rendered,_ = render_segments(x,pieces,100)
            origin = round(placement*100)
            np.testing.assert_array_equal(rendered[origin:],x)
            self.assertFalse(np.any(rendered[:origin]))
            times,valid = pitch_mapping(pieces,plan,len(rendered))
            frame = origin+105
            self.assertTrue(valid[frame])
            self.assertFalse(valid[origin+102])
            self.assertFalse(valid[origin+109])
            self.assertAlmostEqual(times[frame],1.03+offset+.028+max(0.,-offset*.5))

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
        cache=PreviewCache(dict(options={'declip':False},rhythm_plan=plan),{})
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
