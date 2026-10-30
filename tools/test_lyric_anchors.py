"""歌词锚点的缺词、含混与繁简匹配回归。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadowvox.lyric_anchors import match_words, _recognize, _needs_phrase_recognition


def words(text, start=0., step=.3):
    return [(start + i * step, char) for i, char in enumerate(text)]


class LyricAnchorTests(unittest.TestCase):
    def test_repeated_low_diversity_text_is_not_lyric_evidence(self):
        self.assertEqual(match_words(words("词曲词曲词曲词曲词曲"), words("词曲词曲词曲词曲词曲")), [])

    def test_uncertain_recognition_segments_and_words_are_rejected(self):
        sr = 16000
        audio = np.full(4 * sr, .2, dtype=np.float32)

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return {"language": "zh", "segments": [
                    {"avg_logprob": -1.5, "words": [{"start": .1, "end": .4, "word": "假"}]},
                    {"compression_ratio": 3., "words": [{"start": .5, "end": .8, "word": "假"}]},
                    {"no_speech_prob": .9, "words": [{"start": .9, "end": 1.2, "word": "假"}]},
                    {"words": [{"start": 1.3, "end": 1.6, "word": "假", "probability": .4},
                               {"start": 1.7, "end": 1.7, "word": "假", "probability": .9},
                               {"start": 2., "end": 2.3, "word": "真", "probability": .9}]},
                    {"no_speech_prob": .9, "avg_logprob": -.4, "words": [
                        {"start": 2.5, "end": 2.8, "word": "实", "probability": .9}]},
                ]}

        result, _ = _recognize(DummyModel(), audio, sr)
        self.assertEqual([word[1] for word in result], ["真", "实"])

    def test_missing_words_in_singer(self):
        text = "今天我们一起去公园看花明天晚上回家吃饭再见"
        ref = words(text)
        dry_text = text[:7] + text[10:]
        anchors = match_words(words(dry_text, start=10.), ref, offset=-10.)
        self.assertGreaterEqual(len(anchors), 3)
        # 漏掉三字后，后续锚点指向原唱对应的位置，而不是提前的下一句。
        later = [(a, b) for a, b in anchors if a > 13.]
        self.assertTrue(later)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), .9, places=2)

    def test_missing_words_in_reference(self):
        text = "今天我们一起去公园看花明天晚上回家吃饭再见"
        dry = words(text, start=10.)
        ref = words(text[:7] + text[10:])
        anchors = match_words(dry, ref, offset=-10.)
        later = [(a, b) for a, b in anchors if a > 14.]
        self.assertTrue(later)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), -.9, places=2)

    def test_unclear_and_isolated_matches(self):
        self.assertEqual(match_words(words("甲乙"), words("甲乙")), [])
        dry = words("今天我们一起去哈哈哈明天晚上回家吃饭再见", start=10.)
        ref = words("今天我们一起去公园看明天晚上回家吃饭再见")
        anchors = match_words(dry, ref, offset=-10.)
        self.assertGreaterEqual(len(anchors), 3)
        self.assertAlmostEqual(anchors[-1][1] - anchors[-1][0], -10., places=2)

    def test_traditional_and_simplified(self):
        dry = words("中国人民我们一起回家", start=5.)
        ref = words("中國人民我們一起回家")
        anchors = match_words(dry, ref, offset=-5.)
        self.assertGreaterEqual(len(anchors), 2)
        self.assertTrue(all(abs((a - 5.) - b) < 1e-8 for a, b in anchors))

    def test_multiple_missing_lines_keep_later_cluster(self):
        text = "今天我们一起去公园看花明天晚上回家吃饭再见春天来到山上开满鲜花"
        dry_text = text[:7] + text[19:]
        anchors = match_words(words(dry_text, start=10.), words(text), offset=-10.)
        later = [(a, b) for a, b in anchors if a >= 12.]
        self.assertGreaterEqual(len(later), 3)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), 3.6, places=2)

    def test_recognizer_words_in_silence_are_not_anchors(self):
        sr = 16000
        audio = np.zeros(4 * sr, dtype=np.float32)
        audio[sr:2 * sr] = .2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return {"language": "zh", "segments": [{"words": [
                    {"start": .1, "end": .3, "word": "假", "probability": .9},
                    {"start": 1.2, "end": 1.4, "word": "真", "probability": .9},
                    {"start": 2.8, "end": 3., "word": "假", "probability": .9},
                ]}]}

        result, _ = _recognize(DummyModel(), audio, sr)
        self.assertEqual([word[1] for word in result], ["真"])

    def test_english_words_do_not_match_only_by_initial(self):
        dry = "love me baby stay forever now again".split()
        ref = "like more blue sing freely next away".split()
        self.assertEqual(match_words(words(dry), words(ref)), [])

    def test_slow_english_words_still_form_anchors(self):
        text = "hold me close forever through the night together".split()
        dry = [(10. + i * 2.2, word, "en") for i, word in enumerate(text)]
        ref = [(i * 2.2, word, "en") for i, word in enumerate(text)]
        anchors = match_words(dry, ref, offset=-10.)
        self.assertGreaterEqual(len(anchors), 3)
        self.assertTrue(all(abs(a - b - 10.) < 1e-6 for a, b in anchors))

    def test_recognizer_keeps_mixed_scripts_and_apostrophes(self):
        sr = 16000
        audio = np.full(3 * sr, .2, dtype=np.float32)

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return {"language": "ko", "segments": [{"words": [
                    {"start": .2, "end": 1.2, "word": "사랑해 my love", "probability": .9},
                    {"start": 1.3, "end": 2.2, "word": "君を愛してる don't", "probability": .9},
                ]}]}

        result, _ = _recognize(DummyModel(), audio, sr)
        self.assertEqual([word[1] for word in result],
                         ["사", "랑", "해", "my", "love", "君", "を", "愛", "し", "て", "る", "don't"])

    def test_english_subwords_in_japanese_window_join_into_words(self):
        sr = 16000
        audio = np.full(3 * sr, .2, dtype=np.float32)

        class DummyModel:
            def transcribe(self, *_args, **_kwargs):
                return {"language": "ja", "segments": [{"words": [
                    {"start": .2, "end": .4, "word": " won", "probability": .9},
                    {"start": .4, "end": .6, "word": "der", "probability": .9},
                    {"start": .6, "end": .8, "word": "ful", "probability": .9},
                    {"start": 1., "end": 1.5, "word": " night", "probability": .9},
                ]}]}

        result, _ = _recognize(DummyModel(), audio, sr)
        self.assertEqual([word[1] for word in result], ["wonderful", "night"])

    def test_japanese_kanji_do_not_use_mandarin_homophones(self):
        dry = [(i * .3, char, "ja") for i, char in enumerate("君心海君心海")]
        ref = [(i * .3, char, "ja") for i, char in enumerate("軍新孩軍新孩")]
        self.assertEqual(match_words(dry, ref), [])

    def test_equal_han_matches_when_window_languages_differ(self):
        text = "愛心春風明天"
        dry = [(i * .3, char, "ja") for i, char in enumerate(text)]
        ref = [(i * .3, char, "zh") for i, char in enumerate(text)]
        self.assertTrue(match_words(dry, ref))

    def test_japanese_missing_syllables_resume(self):
        text = "君を愛してるいつまでも忘れない君と歩く"
        ref = [(i * .3, char, "ja") for i, char in enumerate(text)]
        dry_text = text[:6] + text[9:]
        dry = [(10. + i * .3, char, "ja") for i, char in enumerate(dry_text)]
        anchors = match_words(dry, ref, offset=-10.)
        later = [(a, b) for a, b in anchors if a >= 13.]
        self.assertTrue(later)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), .9, places=2)

    def test_chinese_characters_do_not_match_latin_pinyin_words(self):
        dry = words("我爱你我爱你")
        ref = words(["wo", "ai", "ni", "wo", "ai", "ni"])
        self.assertEqual(match_words(dry, ref), [])

    def test_multilingual_evidence_triggers_phrase_recognition(self):
        korean = [(i * .3, char, "ko") for i, char in enumerate("사랑해항상내곁에")]
        english = [(3. + i * .3, word, "ko") for i, word in enumerate("my love forever tonight".split())]
        self.assertTrue(_needs_phrase_recognition(korean + english, korean + english))

    def test_isolated_foreign_tokens_do_not_force_chinese_phrase_recognition(self):
        chinese = [(i * .3, char, "zh") for i, char in enumerate("今天我们一起去公园看花明天晚上回家吃饭")]
        noise = [(10. + i * .3, word, "zh") for i, word in enumerate(["oh", "yeah", "hey"])]
        self.assertFalse(_needs_phrase_recognition(chinese + noise, chinese))

    def test_rare_foreign_hallucinations_do_not_override_long_chinese_song(self):
        chinese = [(i * .3, char, "zh") for i, char in enumerate("今天我们一起唱歌" * 80)]
        noise = [(i * 20., word, "zh") for i, word in enumerate(["oh", "yeah", "hey", "bye"])]
        self.assertFalse(_needs_phrase_recognition(chinese + noise, chinese))

    def test_korean_english_missing_words_resume(self):
        text = ["사", "랑", "해", "my", "love", "항", "상", "내", "곁", "에", "you", "and", "me", "오", "늘"]
        ref = [(i * .3, word, "ko") for i, word in enumerate(text)]
        dry = [(10. + i * .3, word, "ko") for i, word in enumerate(text[:5] + text[8:])]
        anchors = match_words(dry, ref, offset=-10.)
        later = [(a, b) for a, b in anchors if a >= 12.]
        self.assertTrue(later)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), .9, places=2)

    def test_korean_english_reference_missing_words_resume(self):
        text = ["사", "랑", "해", "my", "love", "항", "상", "내", "곁", "에", "you", "and", "me", "오", "늘"]
        dry = [(10. + i * .3, word, "ko") for i, word in enumerate(text)]
        ref = [(i * .3, word, "ko") for i, word in enumerate(text[:5] + text[8:])]
        anchors = match_words(dry, ref, offset=-10.)
        later = [(a, b) for a, b in anchors if a >= 13.]
        self.assertTrue(later)
        self.assertAlmostEqual(later[-1][1] - (later[-1][0] - 10.), -.9, places=2)

    def test_overlapping_windows_detect_language_and_do_not_duplicate_words(self):
        sr = 16000
        seconds = 60
        audio = (.2 + np.arange(seconds * sr, dtype=np.float32) / sr * .01)
        events = [(19.8, "hello"), (20.4, "사랑해"), (40.2, "君を愛してる")]

        class DummyModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, x, **kwargs):
                start = round((float(x[0]) - .2) * 100, 2)
                self.calls.append(kwargs.get("language"))
                lang = "en" if start < 1 else "ko" if start < 30 else "ja"
                return {"language": lang, "segments": [{"words": [
                    {"start": t - start, "end": t - start + .2, "word": word, "probability": .9}
                    for t, word in events if start <= t and t + .2 <= start + len(x) / sr
                ]}]}

        model = DummyModel()
        result, _ = _recognize(model, audio, sr)
        self.assertGreater(len(model.calls), 1)
        self.assertTrue(all(lang is None for lang in model.calls))
        self.assertEqual([word[1] for word in result], ["hello", "사", "랑", "해", "君", "を", "愛", "し", "て", "る"])
        self.assertAlmostEqual(result[0][0], 19.8, delta=.05)

    def test_language_is_redetected_at_clear_phrase_breaks(self):
        sr = 16000
        audio = np.full(18 * sr, .2, dtype=np.float32)
        for start in (4, 9, 14):
            audio[start * sr:(start + 1) * sr] = 0

        class DummyModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, x, **kwargs):
                self.calls.append((len(x) / sr, kwargs.get("language")))
                return {"language": "ko", "segments": []}

        model = DummyModel()
        _recognize(model, audio, sr)
        self.assertGreaterEqual(len(model.calls), 3)
        self.assertTrue(all(language is None for _, language in model.calls))


if __name__ == "__main__":
    unittest.main()
