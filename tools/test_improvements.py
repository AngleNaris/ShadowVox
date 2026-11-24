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
        self.assertGreaterEqual(rap.mix_duck, 1.3)
        self.assertTrue(rap.mix_effects)

        rock = jobs.options_from({'genre_preset': '摇滚'})
        self.assertGreaterEqual(rock.strength, 0.75)
        self.assertGreaterEqual(rock.mix_duck, 1.3)
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

        dance = jobs.options_from({'genre_preset': '舞曲'})
        self.assertGreaterEqual(dance.strength, 0.9)
        self.assertGreaterEqual(dance.register_detail, 0.2)
        self.assertEqual(dance.large_note_limit_cents, 380.0)
        self.assertEqual(dance.target_lufs, -8.0)

    def test_genre_preset_preserves_dry_has_backing_flag(self):
        opts = jobs.options_from({'genre_preset': '民谣', 'dry_has_backing': True})
        self.assertTrue(opts.dry_has_backing)
        self.assertEqual(opts.genre_preset, '民谣')


if __name__ == '__main__':
    unittest.main()
