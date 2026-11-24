"""覆盖本次全部对齐、音高、美化/母带改进与独立曲风配方的回归测试。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from shadowvox import full_align, hubert_alignment, jobs, mixing, presets, rhythm, signalsmith
from shadowvox.align import estimate_shift
from shadowvox.f0 import FcpeExtractor, from_cent


class AlignmentImprovementTests(unittest.TestCase):
    def test_oov_pinyin_is_normalized_instead_of_dropping_window(self):
        aligner = object.__new__(hubert_alignment.ChineseAligner)
        aligner.sr = 16000
        aligner.hop = 320
        aligner.vocab = {}
        class FakeG2P:
            dictionary = {'ni': ['n', 'i'], 'hao': ['h', 'ao'], 'en': ['en']}
            def __call__(self, text):
                toks = text.split()
                return ['SP'] + toks + ['SP'], toks, [0] * (len(toks) + 2)

        aligner.g2p = FakeG2P()
        aligner.session = SimpleNamespace(run=lambda *_: (np.zeros((1, 4, 10)), np.zeros((1, 10))))
        fake_word = lambda tok, i: SimpleNamespace(
            text=tok, start=i * .2, end=(i + 1) * .2,
            phonemes=[SimpleNamespace(start=i * .2 + .05, end=(i + 1) * .2)],
        )
        with patch.object(hubert_alignment, 'AlignmentDecoder') as dec:
            dec.return_value.decode.side_effect = lambda *args: (
                [fake_word(tok, i) for i, tok in enumerate(args[4])], 0.9
            )
            out = aligner.decode(np.zeros(16000, dtype=np.float32), 16000, '嗯你好')
        self.assertEqual([w['text'] for w in out], ['en', 'ni', 'hao'])

    def test_pause_rendering_preserves_edge_breath_without_pitch_shift(self):
        sr = 8000
        t = np.arange(round(.12 * sr)) / sr
        # 在停顿两端放置 440Hz 模拟换气/尾音，中间为安静区。
        x = np.zeros(round(.6 * sr), dtype=np.float32)
        breath = (.08 * np.sin(2 * np.pi * 440 * t) * np.hanning(len(t))).astype(np.float32)
        x[:len(breath)] = breath
        x[-len(breath):] = breath
        for want in (round(.4 * sr), round(.85 * sr)):
            with self.subTest(want=want):
                y = full_align._render_pause(x, want, sr)
                self.assertEqual(len(y), want)
                # 首尾 80ms 的波形保持原速原调（逐采样一致）。
                edge = round(.08 * sr)
                np.testing.assert_allclose(y[:edge], x[:edge], atol=1e-6)
                np.testing.assert_allclose(y[-edge:], x[-edge:], atol=1e-6)

    def test_wide_onset_offset_recovers_large_recording_delay(self):
        # 构造 16 秒旋律（每 0.6s 变一次音高），干声整体延后 3.4 秒（超过旧 ±1.5s 上限）。
        notes = [220, 262, 294, 330, 392, 330, 294, 247, 220, 294, 349, 392, 440, 392, 330, 262]
        ref = np.zeros(1600, dtype=np.float32)
        for i, hz in enumerate(notes):
            a = 100 + i * 75
            ref[a:a + 50] = hz
        delay_frames = 340
        dry = np.r_[np.zeros(delay_frames, dtype=np.float32), ref[:-delay_frames]]
        est = rhythm.wide_onset_offset(dry, ref)
        self.assertAlmostEqual(est, 3.4, delta=.05)

    def test_acoustic_onset_groups_fill_unmatched_phrases_and_reject_wrong_melody(self):
        ref = np.zeros(800, dtype=np.float32)
        dry = np.zeros(800, dtype=np.float32)
        # 第 1 句（1.0s 起）：4 个同旋律起音，干声分别慢 80ms / 90ms / 70ms / 85ms
        for t_ref, delay, hz in ((1.0, .08, 260.), (1.5, .09, 294.), (2.0, .07, 330.), (2.5, .085, 349.)):
            r0 = int(round(t_ref * 100))
            d0 = int(round((t_ref + delay) * 100))
            ref[r0:r0 + 25] = hz
            dry[d0:d0 + 25] = hz * 1.03  # 略偏 50 音分但同旋律
        # 第 2 句（5.0s 起）：干声与原唱唱完全不同旋律（偏 500 音分），应被拒绝
        for t_ref in (5.0, 5.5, 6.0):
            r0 = int(round(t_ref * 100))
            ref[r0:r0 + 25] = 260.
            dry[r0 + 8:r0 + 33] = 260. * (2 ** (5. / 12.))
        extra = hubert_alignment._acoustic_onset_groups(dry, ref, 0.0, [], 8.0, 8.0)
        self.assertEqual(len(extra), 1)
        self.assertEqual(len(extra[0][2]), 4)
        self.assertAlmostEqual(extra[0][2][0][0]['start'] - extra[0][2][0][1]['start'], 0.08, delta=.015)

    def test_wide_onset_offset_and_affine_estimator_handle_transposed_large_offset(self):
        # 构造 90 秒旋律：干声升 +3 半音，且比原唱早 4.2 秒（ref - dry = +4.2s）
        rng = np.random.default_rng(11)
        notes = [220, 247, 262, 294, 330, 349, 392, 440]
        ref = np.zeros(9500, dtype=np.float32)
        dry = np.zeros(9500, dtype=np.float32)
        t = 1.0
        while t < 84.0:
            hz = float(notes[int(rng.integers(0, len(notes)))])
            dur = float(rng.uniform(0.28, 0.55))
            d0, d1 = int(round(t * 100)), int(round((t + dur) * 100))
            r0, r1 = d0 + 420, d1 + 420
            dry[d0:d1] = hz * (2.0 ** (3.0 / 12.0))
            ref[r0:r1] = hz
            t += dur + float(rng.uniform(0.18, 0.42))
        wo = rhythm.wide_onset_offset(dry, ref)
        self.assertAlmostEqual(-wo, 4.2, delta=.06)
        aff = rhythm.estimate_affine_alignment(dry, ref)
        self.assertTrue(aff['reliable'])
        self.assertFalse(aff['scale_applied'])
        self.assertAlmostEqual(aff['offset'], 4.2, delta=.05)
        self.assertEqual(aff['key_semitones'], 3)

    def test_affine_alignment_recovers_linear_tempo_drift_and_maps_coordinates(self):
        # 构造 120 秒旋律：干声比原唱快 0.5%（scale = 1.005），且首端偏移 -0.18s（t=0 时 ref 比 dry 早 180ms）
        rng = np.random.default_rng(23)
        notes = [220, 262, 294, 330, 349, 392, 440, 494]
        true_scale, true_offset = 1.005, -0.18
        ref = np.zeros(12500, dtype=np.float32)
        dry = np.zeros(12500, dtype=np.float32)
        t = 2.0
        while t < 116.0:
            hz = float(notes[int(rng.integers(0, len(notes)))])
            dur = float(rng.uniform(0.25, 0.50))
            d0, d1 = int(round(t * 100)), int(round((t + dur) * 100))
            tr = true_scale * t + true_offset
            r0, r1 = int(round(tr * 100)), int(round((tr + dur * true_scale) * 100))
            if r0 >= 0 and r1 < len(ref):
                dry[d0:d1] = hz
                ref[r0:r1] = hz
            t += dur + float(rng.uniform(0.16, 0.38))
        aff = rhythm.estimate_affine_alignment(dry, ref)
        self.assertTrue(aff['reliable'])
        self.assertTrue(aff['scale_applied'])
        self.assertAlmostEqual(aff['scale'], true_scale, delta=.0006)
        self.assertAlmostEqual(aff['offset'], true_offset, delta=.04)

        # 验证 hubert_alignment 全链路坐标（word_groups → build_plan → plan_segments → pitch_mapping）
        def mk(text, s):
            return dict(text=text, start=s, end=s + .28, final_start=s + .06, final_end=s + .28)
        left = [mk(w, 20. + i * .35) for i, w in enumerate(('wo', 'ai', 'ni', 'hao'))] + \
               [mk(w, 100. + i * .35) for i, w in enumerate(('jin', 'tian', 'zhen', 'bang'))]
        right = [mk(w['text'], true_scale * w['start'] + true_offset) for w in left]
        groups = [(0., 120., g) for g in hubert_alignment.word_groups(left, right, true_offset, scale=true_scale)]
        self.assertEqual(sum(len(g) for _, _, g in groups), 8)
        voiced = np.ones(1200, bool)
        plan = hubert_alignment.build_plan(groups, voiced, np.ones(1250, bool), 120., 125., true_offset, scale=true_scale)
        pieces, stats = hubert_alignment.plan_segments(plan, strength=1.0, max_rate=0.10)
        self.assertAlmostEqual(stats['global_tempo_scale'], true_scale, places=5)
        self.assertAlmostEqual(stats['reference_shift_seconds'], -true_offset, places=5)
        self.assertGreaterEqual(stats['rate_min'], 0.9 - 1e-6)
        self.assertLessEqual(stats['rate_max'], 1.1 + 1e-6)
        # 验证输出时间轴与 shifted reference 时间轴精确对齐
        src_all = np.concatenate([p['source'][:-1] for p in pieces] + [pieces[-1]['source'][-1:]])
        dst_all = np.concatenate([p['target'][:-1] for p in pieces] + [pieces[-1]['target'][-1:]])
        for s_t in (20.06, 100.06):
            expected_out = true_scale * s_t + true_offset + stats['reference_shift_seconds']
            self.assertAlmostEqual(float(np.interp(s_t, src_all, dst_all)), expected_out, delta=.005)
        # 验证 pitch_mapping 反向映射精确命中原唱时间
        ref_times, valid = full_align.pitch_mapping(pieces, plan, 12500, 100.)
        frame_late = int(round((true_scale * 100.12 + true_offset + stats['reference_shift_seconds']) * 100))
        self.assertTrue(valid[frame_late])
        self.assertAlmostEqual(ref_times[frame_late], true_scale * 100.12 + true_offset + stats['reference_shift_seconds'], delta=.02)

    def test_confirmed_affine_accepts_strong_acoustic_consensus_without_lyrics(self):
        acoustic = dict(scale=1.0, offset=-0.202, reliable=True, scale_applied=False,
                        inliers=18, total_windows=19, span_seconds=170.0, const_residual_ms=9.5)
        scale, off, ok = hubert_alignment._confirmed_affine([], acoustic, -0.202)
        self.assertTrue(ok)
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(off, -0.202)
        # 不可靠声学估计不得误触发
        weak = dict(scale=1.0, offset=-0.202, reliable=False, scale_applied=False,
                    inliers=2, total_windows=10, span_seconds=15.0, const_residual_ms=90.0)
        self.assertEqual(hubert_alignment._confirmed_affine([], weak, -0.202), (1.0, 0.0, False))

    def test_phrase_timeline_keeps_valid_stretch_when_stft_end_phase_rotates(self):
        sr = 8000
        t = np.arange(4 * sr) / sr
        dry = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        pieces = [
            dict(source=np.array([0., 1.]), target=np.array([0., 1.]), pause=True, phrase_move=0., protected=True, phrase_timeline=True),
            dict(source=np.array([1., 2., 3.]), target=np.array([1., 2.04, 3.]), pause=False, phrase_move=0., protected=False, phrase_timeline=True),
            dict(source=np.array([3., 4.]), target=np.array([3., 4.]), pause=True, phrase_move=0., protected=True, phrase_timeline=True),
        ]
        # 模拟 Signalsmith 正常长度输出但右边界相位旋转（反相）
        with patch('shadowvox.full_align.stretch', side_effect=lambda x, *_, **__: -x):
            out, _ = full_align.render_segments(dry, pieces, sr)
        self.assertFalse(any('render_fallback' in p for p in pieces))
        self.assertEqual(len(out), len(dry))
        # 中段保持拉伸输出（未被丢弃回退为原干声）
        mid = out[int(1.5 * sr):int(2.5 * sr)]
        np.testing.assert_allclose(mid, -dry[int(1.5 * sr):int(2.5 * sr)], atol=1e-6)


class PitchAndMixingImprovementTests(unittest.TestCase):
    def test_quiet_recording_f0_is_not_dropped_by_gate(self):
        sr = 16000
        t = np.arange(sr) / sr
        quiet = (0.002 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # 低于 -50 dBFS 门限
        captured = {}
        ext = object.__new__(FcpeExtractor)
        ext.in_name = 'mel'
        ext.sess = SimpleNamespace(run=lambda _, feed: (captured.setdefault('mel', feed['mel']), [np.zeros((1, feed['mel'].shape[1], 360), dtype=np.float32)])[1])
        ext.extract(quiet, sr)
        # 归一化后送入梅尔谱的能量应明显高于静音底噪 (-11.51)
        self.assertGreater(float(np.mean(captured['mel'])), -8.0)

    def test_signalsmith_smooths_step_shifts_without_phase_cancellation_dips(self):
        sr = 44100
        t = np.arange(sr) / sr
        x = (0.25 * np.sin(2 * np.pi * 260 * t) + 0.10 * np.sin(2 * np.pi * 520 * t)).astype(np.float32)
        shift = np.zeros(100)
        shift[30:70] = 150.0
        f0 = np.full(100, 260.0, dtype=np.float32)
        smooth = signalsmith.smooth_render_shift(shift, f0=f0)
        self.assertLessEqual(float(np.max(np.abs(np.diff(smooth)))), 40.01)
        self.assertAlmostEqual(float(np.median(smooth[40:60])), 150.0, delta=2.0)
        y = signalsmith.render_pitch(x, smooth, sr, f0=f0)
        hop = int(0.02 * sr)
        env_x = np.sqrt(np.mean(x[:len(x) // hop * hop].reshape(-1, hop) ** 2, axis=1))
        env_y = np.sqrt(np.mean(y[:len(y) // hop * hop].reshape(-1, hop) ** 2, axis=1))
        # 不得出现干湿相位抵消导致的明显包络凹坑
        self.assertGreater(float(np.min(env_y / np.maximum(env_x, 1e-8))), 0.80)

    def test_vocal_gain_db_is_true_post_fader_without_altering_compression_or_ducking(self):
        sr = 16000
        t = np.arange(4 * sr) / sr
        voice = (0.18 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        bed = np.stack([0.12 * np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32)
        _, stats0 = mixing.enhance(voice, bed, sr, vocal_gain_db=0.)
        _, stats6 = mixing.enhance(voice * (10 ** (6. / 20.)), bed, sr, vocal_gain_db=6.)
        self.assertAlmostEqual(stats0['max_compression_db'], stats6['max_compression_db'], places=5)
        self.assertAlmostEqual(stats0['max_band_duck_db'], stats6['max_band_duck_db'], places=5)
        self.assertAlmostEqual(stats0['max_deess_db'], stats6['max_deess_db'], places=5)

    def test_custom_large_note_limit_allows_dance_style_wide_correction(self):
        target = np.full(160, 5400.)
        source = target + 350.
        default_res = estimate_shift(from_cent(source), from_cent(target), strength=1., pitch_mode='register',
                                     offset_override=0., reference_times=np.arange(160) / 100.,
                                     reference_valid=np.ones(160, bool))
        wide_res = estimate_shift(from_cent(source), from_cent(target), strength=1., pitch_mode='register',
                                  offset_override=0., reference_times=np.arange(160) / 100.,
                                  reference_valid=np.ones(160, bool), large_note_limit_cents=400.)
        np.testing.assert_array_equal(default_res.shift_cents, 0.)
        np.testing.assert_allclose(wide_res.shift_cents[10:-10], -350., atol=.5)

    def test_acoustic_pitch_fallback_recovers_missed_lyrics_but_rejects_wrong_melody(self):
        from shadowvox.align import augment_pitch_correspondence
        ref = np.zeros(600, dtype=np.float32)
        dry = np.zeros(600, dtype=np.float32)
        # 第 1 句（50..200）：原唱与干声同旋律，干声走音 +95 音分，但歌词未匹配
        ref[50:200] = from_cent(np.full(150, 5400.))
        dry[50:200] = from_cent(np.full(150, 5495.))
        # 第 2 句（300..450）：干声乱唱完全不同旋律（偏 520 音分），应被拒绝
        ref[300:450] = from_cent(np.full(150, 5400.))
        dry[300:450] = from_cent(np.full(150, 5920.))
        # 第 3 句（500..580）：伴奏纯空档，原唱无声，应被拒绝
        dry[500:580] = from_cent(np.full(80, 5400.))
        times, valid, added = augment_pitch_correspondence(
            dry, ref, np.zeros(600), np.zeros(600, dtype=bool))
        self.assertGreaterEqual(added, 140)
        self.assertTrue(valid[60:190].all())
        self.assertFalse(valid[300:450].any())
        self.assertFalse(valid[500:580].any())


class IndependentGenrePresetTests(unittest.TestCase):
    def test_all_eight_genres_are_decoupled_and_acoustically_distinct(self):
        self.assertEqual(len(presets.GENRE_NAMES), 8)
        rap = jobs.options_from({'genre_preset': '说唱', 'correction': 100, 'mix_effects': False})
        # 说唱：关闭音高修正，但独立保留节奏对齐、降噪、削波修复、空档静音与高伴奏避让！
        self.assertEqual(rap.strength, 0.0)
        self.assertTrue(rap.rhythm and rap.denoise and rap.declip and rap.mute_reference_gaps)
        self.assertEqual(rap.rhythm_strength, 1.0)
        self.assertEqual(rap.denoise_strength, 1.0)
        self.assertLessEqual(rap.mix_reverb, 0.5)
        self.assertGreaterEqual(rap.mix_duck, 0.8)
        self.assertTrue(rap.mix_effects)

        rock = jobs.options_from({'genre_preset': '摇滚'})
        self.assertGreaterEqual(rock.strength, 0.75)
        self.assertGreaterEqual(rock.mix_duck, 0.75)
        self.assertGreater(rock.mix_profile['low_shelf_gain_db'], 0.0)
        self.assertGreater(rock.mix_profile['dynamic_body_max_db'], 0.0)

        folk = jobs.options_from({'genre_preset': '民谣'})
        self.assertEqual(folk.mix_lowcut_hz, 80.0)
        self.assertGreaterEqual(folk.strength, 0.70)
        self.assertLessEqual(folk.target_lufs, -9.0)
        self.assertEqual(folk.true_peak_ceiling, -1.0)

        ballad = jobs.options_from({'genre_preset': '抒情流行'})
        guofeng = jobs.options_from({'genre_preset': '中国风'})
        self.assertGreater(guofeng.mix_profile['reverb_room_size'], ballad.mix_profile['reverb_room_size'])
        self.assertGreater(guofeng.mix_profile['high_shelf_gain_db'], ballad.mix_profile['high_shelf_gain_db'])
        self.assertLessEqual(guofeng.mix_profile['long_room_send'], 1.15)
        self.assertGreaterEqual(guofeng.mix_profile['wet_duck_db'], 9.0)
        for name in presets.GENRE_NAMES:
            g = jobs.options_from({'genre_preset': name})
            self.assertLessEqual(g.vocal_gain_db, 0.0)
            self.assertGreaterEqual(g.mix_profile['wet_duck_db'], 7.5)

        dance = jobs.options_from({'genre_preset': '舞曲'})
        self.assertGreaterEqual(dance.strength, 0.9)
        self.assertGreaterEqual(dance.register_detail, 0.2)
        self.assertEqual(dance.large_note_limit_cents, 380.0)
        self.assertEqual(dance.target_lufs, -8.0)

    def test_genre_preset_preserves_dry_has_backing_flag(self):
        opts = jobs.options_from({'genre_preset': '民谣', 'dry_has_backing': True})
        self.assertTrue(opts.dry_has_backing)
        self.assertEqual(opts.genre_preset, '民谣')


class KeyShiftAndPartialFadeoutTests(unittest.TestCase):
    def test_key_shift_detects_transposed_singing_and_rejects_same_key_or_octave(self):
        from shadowvox.key_shift import detect_key_shift, transpose_f0, transpose_audio
        rng = np.random.default_rng(7)
        # 6 秒旋律（600 帧）
        base_cents = np.repeat(np.array([5000., 5200., 5400., 5700., 5900., 5500.]), 100)
        ref_f0 = from_cent(base_cents)

        # 1) 异调升 +3 半音（带 ±25c 正常演唱波动）应精准识别为 +3st
        dry_plus3 = from_cent(base_cents + 300. + rng.normal(0., 22., len(base_cents)))
        res = detect_key_shift(dry_plus3, ref_f0)
        self.assertTrue(res['applied'])
        self.assertEqual(res['semitones'], 3)
        self.assertGreater(res['before_error_cents'], 250.)
        self.assertLess(res['after_error_cents'], 45.)

        # 2) 纯低八度跟唱（-12 半音）属于八度折叠，不得触发伴奏异调
        dry_oct = from_cent(base_cents - 1200. + rng.normal(0., 20., len(base_cents)))
        self.assertFalse(detect_key_shift(dry_oct, ref_f0)['applied'])

        # 3) 同调偏高 55 音分（普通走音）不得误判为 +1 半音异调
        dry_sharp = from_cent(base_cents + 55. + rng.normal(0., 25., len(base_cents)))
        self.assertFalse(detect_key_shift(dry_sharp, ref_f0)['applied'])

        # 4) F0 与立体声伴奏移调保持长度与目标音高
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        stereo = np.vstack([0.3 * np.sin(2 * np.pi * 440. * t)] * 2).astype(np.float32)
        shifted = transpose_audio(stereo, sr, 3)
        self.assertEqual(shifted.shape, stereo.shape)
        self.assertTrue(np.isfinite(shifted).all())
        shifted_f0 = transpose_f0(ref_f0, 3)
        np.testing.assert_allclose(shifted_f0 / ref_f0, 2.0 ** (3 / 12), rtol=1e-5)

    def test_partial_fadeout_trims_partial_cover_but_keeps_natural_outro(self):
        from shadowvox.ending import plan_partial_fadeout, apply_fadeout
        sr = 8000
        # 原曲 120s，原唱唱到 100s（含 20s 自然纯乐器尾奏）
        ref = np.zeros(120 * sr, dtype=np.float32)
        ref[5 * sr:100 * sr] = 0.2

        # 1) 全曲翻唱（唱到 98s，自然尾奏 22s）不应触发半曲淡出
        full_dry = np.zeros(120 * sr, dtype=np.float32)
        full_dry[5 * sr:98 * sr] = 0.2
        plan_full = plan_partial_fadeout(full_dry, ref, sr)
        self.assertFalse(plan_full['applied'])

        # 2) 半曲翻唱（仅唱到 50s）应触发：50s + 3.5s 缓冲 + 4.5s 淡出 = 58.0s
        part_dry = np.zeros(120 * sr, dtype=np.float32)
        part_dry[5 * sr:50 * sr] = 0.2
        plan_part = plan_partial_fadeout(part_dry, ref, sr)
        self.assertTrue(plan_part['applied'])
        self.assertAlmostEqual(plan_part['last_vocal_seconds'], 50.0, delta=0.2)
        self.assertAlmostEqual(plan_part['end_seconds'], 58.0, delta=0.2)

        # 3) 淡出平滑归零且不改动缓冲前正文
        sig = np.ones((2, plan_part['end_samples']), dtype=np.float32) * 0.5
        faded = apply_fadeout(sig, sr, plan_part['fade_seconds'])
        self.assertAlmostEqual(float(faded[0, 10 * sr]), 0.5, places=5)
        self.assertAlmostEqual(float(faded[0, -1]), 0.0, places=5)
        self.assertTrue(np.all(np.diff(faded[0, -int(4 * sr):]) <= 1e-6))


class AdaptiveMixAndMSDuckTests(unittest.TestCase):
    def test_adaptive_eq_plan_distinguishes_muffled_sharp_and_balanced_vocals(self):
        from shadowvox import mixing

        muffled = {
            'bands': {'low': -2.6, 'mud': -9.4, 'mid': -13.3, 'pres': -20.2, 'sib': -24.4, 'air': -31.9},
            'body_total_db': -1.8, 'low_centroid_hz': 210.0,
            'pres_minus_low_db': -17.6, 'sib_minus_pres_db': -4.2, 'is_low_male': True,
        }
        p_muf = mixing.plan_adaptive_eq(muffled)
        reasons_muf = {r: g for _, g, _, r in p_muf['adaptive_bells']}
        self.assertLess(reasons_muf['de_boom'], -2.5)
        self.assertGreater(reasons_muf['clarity_mid'], 1.5)
        self.assertGreater(reasons_muf['clarity_pres'], 2.5)
        self.assertGreater(p_muf['air_shelf_delta_db'], 1.5)
        self.assertLess(p_muf['low_boost_scale'], 0.25)
        self.assertEqual(p_muf['body_fill_scale'], 0.0)

        sharp = {
            'bands': {'low': -8.3, 'mud': -5.4, 'mid': -10.4, 'pres': -11.2, 'sib': -9.7, 'air': -20.8},
            'body_total_db': -3.6, 'low_centroid_hz': 310.0,
            'pres_minus_low_db': -2.9, 'sib_minus_pres_db': 1.5, 'is_low_male': False,
        }
        p_sharp = mixing.plan_adaptive_eq(sharp)
        reasons_sharp = {r: g for _, g, _, r in p_sharp['adaptive_bells']}
        self.assertLess(reasons_sharp['tame_harsh'], -2.5)
        self.assertLess(p_sharp['high_shelf_scale'], 0.35)
        self.assertGreater(p_sharp['deess_scale'], 1.3)

        balanced = {
            'bands': {'low': -8.5, 'mud': -6.2, 'mid': -6.0, 'pres': -11.0, 'sib': -17.5, 'air': -21.5},
            'body_total_db': -4.2, 'low_centroid_hz': 285.0,
            'pres_minus_low_db': -2.5, 'sib_minus_pres_db': -6.5, 'is_low_male': False,
        }
        p_bal = mixing.plan_adaptive_eq(balanced)
        self.assertEqual(p_bal['adaptive_bells'], [])
        self.assertEqual(p_bal['air_shelf_delta_db'], 0.0)

    def test_ms_duck_carves_mid_preserves_side_and_shifts_band_for_low_male(self):
        from shadowvox import mixing

        sr = 44100
        t = np.arange(4 * sr, dtype=np.float32) / sr
        active = (t >= 1.0) & (t < 3.0)
        voice = (0.25 * np.sin(2 * np.pi * 180 * t) + 0.05 * np.sin(2 * np.pi * 2000 * t)) * active
        voice = voice.astype(np.float32)
        mid_sig = (0.20 * np.sin(2 * np.pi * 200 * t) + 0.20 * np.sin(2 * np.pi * 2000 * t)
                   + 0.15 * np.sin(2 * np.pi * 65 * t)).astype(np.float32)
        side_sig = (0.18 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        bed = np.vstack([mid_sig + side_sig, mid_sig - side_sig]).astype(np.float32)

        bypassed, bstats = mixing._backing_ms_duck(bed, voice, sr, 0.0, diag={'is_low_male': True})
        np.testing.assert_array_equal(bypassed, bed)
        self.assertFalse(bstats['enabled'])

        ducked, stats = mixing._backing_ms_duck(bed, voice, sr, 0.6, diag={'is_low_male': True})
        self.assertTrue(stats['enabled'])
        self.assertEqual(stats['body_band_hz'], [130.0, 380.0])
        self.assertGreater(stats['mid_pres_duck_db'], 2.4)
        self.assertGreater(stats['mid_body_duck_db'], 2.0)
        self.assertLess(stats['side_pres_duck_db'], stats['mid_pres_duck_db'] * 0.3)

        # Mid 2kHz 明显衰减，Side 2kHz 与 65Hz 超低频保持完整
        win = (t >= 1.5) & (t < 2.5)
        out_mid = 0.5 * (ducked[0] + ducked[1])
        out_side = 0.5 * (ducked[0] - ducked[1])
        tone_2k = np.sin(2 * np.pi * 2000 * t)
        tone_65 = np.sin(2 * np.pi * 65 * t)
        mid_2k_db = 20 * np.log10(abs(np.dot(out_mid[win], tone_2k[win]) / np.dot(mid_sig[win], tone_2k[win])))
        side_2k_db = 20 * np.log10(abs(np.dot(out_side[win], tone_2k[win]) / np.dot(side_sig[win], tone_2k[win])))
        sub_65_db = 20 * np.log10(abs(np.dot(out_mid[win], tone_65[win]) / np.dot(mid_sig[win], tone_65[win])))
        self.assertLess(mid_2k_db, -2.0)
        self.assertGreater(side_2k_db, mid_2k_db + 1.5)
        self.assertAlmostEqual(sub_65_db, 0.0, delta=0.2)

        _, std_stats = mixing._backing_ms_duck(bed, voice, sr, 0.6, diag={'is_low_male': False})
        self.assertEqual(std_stats['body_band_hz'], [220.0, 520.0])

    def test_genre_and_desktop_options_enable_adaptive_mix_and_ms_duck(self):
        for name in presets.GENRE_NAMES:
            opts = jobs.options_from({'genre_preset': name})
            self.assertTrue(opts.mix_profile['adaptive_eq'])
            self.assertTrue(opts.mix_profile['ms_duck'])
            self.assertTrue(opts.mix_profile['perceptual_balance'])
        desk = jobs.options_from(jobs.DEFAULT_PARAMS)
        self.assertTrue(desk.mix_profile['adaptive_eq'])
        self.assertTrue(desk.mix_profile['ms_duck'])
        self.assertTrue(desk.mix_profile['perceptual_balance'])

    def test_female_vocal_gets_higher_highpass_and_low_end_cleanup(self):
        sr = 44100
        t = np.arange(4 * sr, dtype=np.float32) / sr
        # 模拟女声基频 (330Hz) + 泛音，但混入了大量 85Hz 喷麦/低频轰鸣与 310Hz 混浊
        female = (
            0.16 * np.sin(2 * np.pi * 85 * t)
            + 0.14 * np.sin(2 * np.pi * 165 * t)
            + 0.22 * np.sin(2 * np.pi * 330 * t)
            + 0.10 * np.sin(2 * np.pi * 660 * t)
            + 0.05 * np.sin(2 * np.pi * 2200 * t)
        ).astype(np.float32)
        f0 = np.full(400, 330.0, dtype=np.float32)
        bed = np.zeros((2, len(female)), dtype=np.float32)
        opts = jobs.options_from({'genre_preset': '抒情流行'})
        mixed, stats = mixing.enhance(
            female, bed, sr, space=0., duck=0.,
            lowcut_hz=opts.mix_lowcut_hz, eq=opts.mix_eq, compression=opts.mix_compression,
            deessing=opts.mix_deess, harmonics=opts.mix_harmonics,
            profile=opts.mix_profile, vocal_f0=f0,
        )
        self.assertTrue(stats['vocal_diagnosis']['is_female_or_high'])
        self.assertGreaterEqual(stats['effective_lowcut_hz'], 120.0)
        self.assertEqual(stats['highpass_order'], 4)
        # 预设原本的 160Hz +1.4dB 低架应被明显压低，不再叠加低频
        self.assertLessEqual(stats['low_shelf_gain_db'], 0.5)
        # 85Hz 冗余低频相对 330Hz 基频被显著削减（至少降低 12 dB）
        def tone_ratio_db(sig):
            e85 = abs(np.dot(sig[sr:], np.sin(2 * np.pi * 85 * t[sr:])))
            e330 = abs(np.dot(sig[sr:], np.sin(2 * np.pi * 330 * t[sr:])))
            return 20.0 * np.log10(max(e85, 1e-9) / max(e330, 1e-9))
        self.assertLess(tone_ratio_db(mixed[0]) - tone_ratio_db(female), -12.0)

    def test_vocal_leveler_evens_phrase_loudness_without_lifting_noise_floor(self):
        sr = 16000
        t = np.arange(8 * sr, dtype=np.float32) / sr
        # 0-2s 静音底噪(-50dB)，2-5s 轻唱(-27dB)，5-8s 高音爆发(-9dB)
        env = np.where(t < 2.0, 0.003, np.where(t < 5.0, 0.045, 0.35))
        voice = (env * np.sin(2 * np.pi * 260 * t)).astype(np.float32)
        # 先按工作电平归一化后测试 level_vocal
        cal = mixing.WORKING_ACTIVE_RMS / mixing._active_rms(voice, sr)
        work = (voice * cal).astype(np.float32)
        leveled, st = mixing.level_vocal(work, sr, amount=1.0)
        self.assertTrue(st['applied'])
        self.assertGreater(st['max_boost_db'], 2.0)
        self.assertGreater(st['max_cut_db'], 2.5)
        def seg_db(sig, a, b):
            return 20.0 * np.log10(max(float(np.sqrt(np.mean(sig[int(a*sr):int(b*sr)] ** 2))), 1e-9))
        before_gap = seg_db(work, 5.5, 7.5) - seg_db(work, 2.5, 4.5)
        after_gap = seg_db(leveled, 5.5, 7.5) - seg_db(leveled, 2.5, 4.5)
        # 轻唱与强唱之间的响度落差被明显收窄（至少收窄 6 dB）
        self.assertLess(after_gap, before_gap - 6.0)
        # 静音底噪段不得被明显抬升（< 0.5 dB）
        self.assertLess(seg_db(leveled, 0.3, 1.5) - seg_db(work, 0.3, 1.5), 0.5)

    def test_multiband_density_isolates_heavy_bass_from_vocal_midrange(self):
        from shadowvox import density, mastering
        self.assertEqual(density.CROSSOVERS, (220., 1800., 7500.))
        self.assertLessEqual(mastering.MAX_MEDIAN_REDUCTION_DB, 2.2)
        sr = 44100
        t = np.arange(6 * sr, dtype=np.float32) / sr
        # 2-4s 出现强烈 90Hz 低频爆发，全程有稳定 800Hz 人声中频
        bass_env = np.where((t >= 2.0) & (t < 4.0), 0.55, 0.08)
        sig = (bass_env * np.sin(2 * np.pi * 90 * t) + 0.14 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
        stereo = np.vstack([sig, sig])
        out, st = density.densify(stereo, sr, -12.0, amount=0.85)
        self.assertTrue(st['applied'])
        # 低频段独立控峰力度应显著大于中频人声核心段
        p95 = st['band_reduction_p95_db']
        self.assertGreater(p95[0], p95[1] + 0.8)
        # 低频爆发期，800Hz 人声中频几乎不被低频连带压制（变化 < 0.8 dB）
        tone_800 = np.sin(2 * np.pi * 800 * t)
        quiet_win = (t >= 0.5) & (t < 1.5)
        bass_win = (t >= 2.5) & (t < 3.5)
        mid_quiet = abs(np.dot(out[0, quiet_win], tone_800[quiet_win]))
        mid_bass = abs(np.dot(out[0, bass_win], tone_800[bass_win]))
        mid_dip_db = 20.0 * np.log10(max(mid_bass, 1e-9) / max(mid_quiet, 1e-9))
        self.assertGreater(mid_dip_db, -0.8)

    def test_perceptual_balance_keeps_vocal_to_backing_ratio_consistent_across_songs(self):
        from shadowvox import mix_balance
        sr = 16000
        t = np.arange(16 * sr, dtype=np.float32) / sr
        active = ((t % 4.0) < 3.0).astype(np.float32)
        bed = np.vstack([0.15 * np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32)
        # 场景 A：密集原曲（原唱比伴奏低 2.5dB）且翻唱进入时更低（-4dB），应被稳稳拉回到 >= 0dB 区间
        quiet_ref = (0.11 * np.sin(2 * np.pi * 500 * t) * active).astype(np.float32)
        quiet_cov = (0.095 * np.sin(2 * np.pi * 500 * t) * active).astype(np.float32)
        _, gain_a, st_a = mix_balance.curve(
            quiet_cov, np.vstack([quiet_cov, quiet_cov]), bed, quiet_ref, bed, sr, perceptual=True,
        )
        self.assertGreater(st_a['global_gain_db'], 3.0)
        self.assertAlmostEqual(st_a['target_global_ratio_db'], 0.0, delta=0.1)
        # 场景 B：稀疏原曲（原唱比伴奏高 4dB）且翻唱进入时过高（+6dB），应被收敛到 <= +2.2dB 区间
        loud_ref = (0.24 * np.sin(2 * np.pi * 500 * t) * active).astype(np.float32)
        loud_cov = (0.30 * np.sin(2 * np.pi * 500 * t) * active).astype(np.float32)
        _, gain_b, st_b = mix_balance.curve(
            loud_cov, np.vstack([loud_cov, loud_cov]), bed, loud_ref, bed, sr, perceptual=True,
        )
        self.assertLess(st_b['global_gain_db'], -3.0)
        self.assertLessEqual(st_b['target_global_ratio_db'], 2.21)
        # 两个极端场景经全局对齐后，目标比例落差应从原来的 ~6.5 dB 收敛到 <= 2.3 dB
        self.assertLessEqual(abs(st_b['target_global_ratio_db'] - st_a['target_global_ratio_db']), 2.3)


if __name__ == '__main__':
    unittest.main()
