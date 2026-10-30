"""离线识别歌词，为演唱对齐提供可跳词的粗锚点。

识别文本只用于本次匹配，不写入报告。模型结果可能有误，孤立的字不作为锚点。
"""
import math
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
from numba import njit

from . import config as C

WINDOW_SECONDS = 30.
OVERLAP_SECONDS = .5
COARSE_OVERLAP_SECONDS = 3.
MIN_PHRASE_SECONDS = 2.5
FINE_CORE_SECONDS = 8.


def _is_han(char):
    c = ord(char)
    return (0x3400 <= c <= 0x4dbf or 0x4e00 <= c <= 0x9fff or
            0xf900 <= c <= 0xfaff or 0x20000 <= c <= 0x3134f)


def _is_kana(char):
    c = ord(char)
    return 0x3040 <= c <= 0x30ff or 0x31f0 <= c <= 0x31ff or 0x1b000 <= c <= 0x1b16f


def _is_syllable(char):
    c = ord(char)
    return (_is_han(char) or _is_kana(char) or 0x1100 <= c <= 0x11ff or
            0x3130 <= c <= 0x318f or 0xa960 <= c <= 0xa97f or
            0xac00 <= c <= 0xd7ff)


def _tokens(text):
    """中文、假名和韩文按字；其他文字保留完整单词及单词内撇号。"""
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    tokens, word = [], []

    def flush():
        if word:
            tokens.append("".join(word))
            word.clear()

    for i, char in enumerate(text):
        if _is_syllable(char):
            flush()
            tokens.append(char)
        elif unicodedata.category(char)[0] in "LNM":
            word.append(char)
        elif char == "'" and word and i + 1 < len(text) and text[i + 1].isalnum():
            word.append(char)
        else:
            flush()
    flush()
    return tokens


def _join_word_fragments(words):
    """Whisper 在日语等语种中可能将一个英文单词拆成多个相邻片段。"""
    joined = []
    for word in words:
        if word.get("probability", 1.) < .6:
            joined.append(None)
            continue
        text = word["word"]
        previous = joined[-1] if joined else None
        if previous and text and previous["word"] and not text[0].isspace():
            left, right = previous["word"][-1], text[0]
            left_word = unicodedata.category(left)[0] in "LNM" and not _is_syllable(left)
            right_word = unicodedata.category(right)[0] in "LNM" and not _is_syllable(right)
            if (left_word and (right_word or right in "'’")) or (left in "'’" and right_word):
                previous["word"] += text
                previous["end"] = word["end"]
                previous["probability"] = min(previous.get("probability", 1.), word.get("probability", 1.))
                continue
        joined.append(dict(word))
    return [word for word in joined if word is not None]


def _phrase_boundaries(active, start, end):
    """优先在明显的停顿切段；长段再切分，边缘留重叠供识别。"""
    edges = np.diff(np.r_[0, (~active).astype(np.int8), 0])
    gaps = [(a + b) / 20. for a, b in zip(np.flatnonzero(edges == 1),
                                          np.flatnonzero(edges == -1)) if b - a >= 5]
    pauses = [start]
    for gap in gaps:
        if gap - pauses[-1] >= MIN_PHRASE_SECONDS and end - gap >= MIN_PHRASE_SECONDS:
            pauses.append(gap)
    pauses.append(end)
    boundaries = [start]
    for left, right in zip(pauses, pauses[1:]):
        count = math.ceil((right - left) / FINE_CORE_SECONDS)
        boundaries.extend(np.linspace(left, right, count + 1)[1:])
    return boundaries


def _recognize(model, audio, sr, language=None, split_on_pauses=True):
    import torch
    from scipy.signal import resample_poly
    from .full_align import singing

    x = resample_poly(audio, 16000, sr).astype(np.float32)
    active = singing(x, 16000)
    edges = np.diff(np.r_[0, active.astype(np.int8), 0])
    runs = [(a, b) for a, b in zip(np.flatnonzero(edges == 1),
                                    np.flatnonzero(edges == -1)) if b - a >= 5]
    if not runs:
        return [], language
    start = max(0., runs[0][0] / 10. - 1.)
    end = min(len(x) / 16000., runs[-1][1] / 10. + 1.)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if split_on_pauses:
        boundaries = _phrase_boundaries(active, start, end)
        overlap = OVERLAP_SECONDS
    else:
        overlap = COARSE_OVERLAP_SECONDS
        duration = end - start
        count = 1 if duration <= WINDOW_SECONDS else math.ceil(duration / (WINDOW_SECONDS - 2 * overlap))
        boundaries = np.linspace(start, end, count + 1)
    words, first_language = [], language
    for chunk in range(len(boundaries) - 1):
        core_start, core_end = boundaries[chunk:chunk + 2]
        window_start = max(start, core_start - overlap)
        window_end = min(end, core_end + overlap)
        if not active[int(window_start * 10):min(len(active), math.ceil(window_end * 10))].any():
            continue
        result = model.transcribe(x[round(window_start * 16000):round(window_end * 16000)],
                                  language=language, temperature=0, fp16=False,
                                  condition_on_previous_text=False, word_timestamps=True,
                                  verbose=None)
        detected = result.get("language", language)
        if first_language is None:
            first_language = detected
        for segment in result["segments"]:
            logprob = segment.get("avg_logprob")
            # Whisper 的 no_speech_prob 属于整窗；明确的逐词证据可推翻它。
            # 与其官方跳过逻辑一致，不能单靠该值删掉整窗的可靠唱词。
            if ((logprob is not None and logprob < -1.) or
                    (logprob is None and segment.get("no_speech_prob", 0.) > .6) or
                    segment.get("compression_ratio", 0.) > 2.4):
                continue
            japanese = any(_is_kana(char) for word in segment.get("words", []) for char in word["word"])
            for word in _join_word_fragments(segment.get("words", [])):
                if word["end"] <= word["start"]:
                    continue
                middle = window_start + (word["start"] + word["end"]) / 2
                if not core_start <= middle < core_end:
                    continue
                text = _tokens(word["word"])
                for i, token in enumerate(text):
                    t = window_start + word["start"] + (word["end"] - word["start"]) * i / len(text)
                    frame = round(t * 10.)
                    if active[max(0, frame - 2):min(len(active), frame + 3)].any():
                        token_language = "ja" if japanese and len(token) == 1 and _is_han(token) else detected
                        words.append((t, token, token_language, chunk, word.get("probability", 1.)))
    # 重叠处两次识别的词时标可能略有偏移，只去掉跨窗口、近乎同时的相同词。
    words.sort(key=lambda word: word[0])
    unique = []
    for word in words:
        if (unique and word[1] == unique[-1][1] and word[3] != unique[-1][3] and
                word[0] - unique[-1][0] < .2):
            if word[4] > unique[-1][4]:
                unique[-1] = word
        else:
            unique.append(word)
    return [(t, token, lang) for t, token, lang, _, _ in unique], first_language


@lru_cache(maxsize=4096)
def _chinese_phoneme(char):
    from pypinyin import lazy_pinyin, Style

    return lazy_pinyin(char, style=Style.NORMAL)[0].lower()


def _keys(words):
    exact, phonetic = [], []
    for word in words:
        token = unicodedata.normalize("NFKC", word[1]).casefold()
        language = word[2] if len(word) > 2 else "zh"
        exact.append(token)
        phonetic.append(_chinese_phoneme(token) if language == "zh" and len(token) == 1 and
                        _is_han(token) else None)
    return exact, phonetic


def _needs_phrase_recognition(dry_words, ref_words):
    """有较多跨语种文字或识别语种冲突时，再按停顿细分重识别。"""
    def profile(words):
        scripts, languages = Counter(), Counter()
        for _, token, language in words:
            if language:
                languages[language] += 1
            if not token:
                continue
            char = token[0]
            if _is_han(char) or _is_kana(char):
                scripts["cjk"] += 1
            elif _is_syllable(char):
                scripts["hangul"] += 1
            elif char.isalpha():
                scripts[unicodedata.name(char, "OTHER").split()[0]] += 1
        return scripts, languages

    dry_scripts, dry_languages = profile(dry_words)
    ref_scripts, ref_languages = profile(ref_words)
    for scripts, languages in ((dry_scripts, dry_languages), (ref_scripts, ref_languages)):
        script_total = sum(scripts.values())
        language_total = sum(languages.values())
        if sum(count >= 4 and count >= .01 * script_total for count in scripts.values()) >= 2:
            return True
        if sum(count >= 6 and count >= .01 * language_total for count in languages.values()) >= 2:
            return True
    if dry_languages and ref_languages:
        dry_language, dry_count = dry_languages.most_common(1)[0]
        ref_language, ref_count = ref_languages.most_common(1)[0]
        if dry_language != ref_language and dry_count >= 6 and ref_count >= 6:
            return True
    return False


@njit
def _match(a, b, pa, pb, times_a, times_b, offset):
    """单调序列匹配；两侧可分别跳过未识别、漏唱或多唱的字。"""
    n, m = len(a), len(b)
    score = np.empty((n + 1, m + 1), np.float32)
    parent = np.empty((n + 1, m + 1), np.uint8)
    for i in range(n + 1):
        score[i, 0] = -i * .65
        parent[i, 0] = 1
    for j in range(m + 1):
        score[0, j] = -j * .65
        parent[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = a[i - 1] == b[j - 1] or (pa[i - 1] >= 0 and pa[i - 1] == pb[j - 1])
            value = 2. if same else -1.2
            if value > 0:
                value -= min(.8, .025 * max(0., abs(times_b[j - 1] - times_a[i - 1] - offset) - 2.))
            both = score[i - 1, j - 1] + value
            skip_a = score[i - 1, j] - .65
            skip_b = score[i, j - 1] - .65
            if skip_a > both and skip_a >= skip_b:
                score[i, j], parent[i, j] = skip_a, 1
            elif skip_b > both:
                score[i, j], parent[i, j] = skip_b, 2
            else:
                score[i, j], parent[i, j] = both, 0
    i, j = n, m
    left, right = [], []
    while i > 0 and j > 0:
        step = parent[i, j]
        if step == 0:
            if a[i - 1] == b[j - 1] or (pa[i - 1] >= 0 and pa[i - 1] == pb[j - 1]):
                left.append(i - 1)
                right.append(j - 1)
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return left[::-1], right[::-1]


def match_words(dry_words, ref_words, offset=0.):
    """返回有相邻字支持的单调锚点，避免用单个偶然同音字定位。"""
    if not dry_words or not ref_words:
        return []
    left, left_phonetic = _keys(dry_words)
    right, right_phonetic = _keys(ref_words)
    vocabulary = {word: i for i, word in enumerate(set(left + right))}
    a = np.array([vocabulary[word] for word in left], dtype=np.int32)
    b = np.array([vocabulary[word] for word in right], dtype=np.int32)
    phonetics = {word: i for i, word in enumerate(set(left_phonetic + right_phonetic) - {None})}
    pa = np.array([phonetics.get(word, -1) for word in left_phonetic], dtype=np.int32)
    pb = np.array([phonetics.get(word, -1) for word in right_phonetic], dtype=np.int32)
    da = np.array([word[0] for word in dry_words], dtype=np.float64)
    rb = np.array([word[0] for word in ref_words], dtype=np.float64)
    ia, ib = _match(a, b, pa, pb, da, rb, offset)
    pairs = [(float(da[i]), float(rb[j]), left[i]) for i, j in zip(ia, ib)]
    supported = []
    for a_time, b_time, token in pairs:
        spaced_word = not (len(token) == 1 and _is_syllable(token))
        a_window, b_window = (4.8, 6.) if spaced_word else (2., 3.)
        nearby = [(x, y, text) for x, y, text in pairs
                  if abs(a_time - x) <= a_window and abs(b_time - y) <= b_window]
        # 低密度单词可使用更长的上下文，但不能只靠同一个反复出现的词。
        if len(nearby) >= 3 and len({text for _, _, text in nearby}) >= (2 if spaced_word else 3):
            supported.append((a_time, b_time, spaced_word))
    # 相邻窗口只选一个中间字，减轻识别器逐字时间的抖动。
    anchors = []
    i = 0
    while i < len(supported):
        group = []
        start = supported[i][0]
        window = 4.8 if supported[i][2] else 1.5
        while i < len(supported) and supported[i][0] < start + window:
            group.append(supported[i])
            i += 1
        if len(group) < 2:
            continue
        # 首个受支持的字与组中间的字都可能限定开头，允许分别作为锚点。
        candidates = ([group[0]] if not anchors else []) + [group[len(group) // 2]]
        for candidate in candidates:
            anchor = candidate[:2]
            if not anchors or (anchor[0] - anchors[-1][0] >= .8 and
                               anchor[1] - anchors[-1][1] >= .4):
                anchors.append(anchor)
    # 偶然识别出的同音字可能跳到另一段重复歌词。偏离整体位置较远时，
    # 至少要有一组相邻锚点共同支持，才能将其视为真实漏句/结构变化。
    reliable = []
    for a_time, b_time in anchors:
        delta = b_time - a_time
        nearby = sum(abs(x - a_time) <= 15. and abs((y - x) - delta) <= 1.
                     for x, y in anchors)
        if abs(delta - offset) <= 3. or nearby >= 3:
            reliable.append((a_time, b_time))
    return reliable


def find_anchors(dry, reference, sr, offset=0.):
    import whisper

    model = whisper.load_model(str(C.MODELS / "whisper-base.pt"), device="cpu")
    ref_words, _ = _recognize(model, reference, sr, split_on_pauses=False)
    dry_words, _ = _recognize(model, dry, sr, split_on_pauses=False)
    if _needs_phrase_recognition(dry_words, ref_words):
        ref_words, _ = _recognize(model, reference, sr)
        dry_words, _ = _recognize(model, dry, sr)
    return match_words(dry_words, ref_words, offset)
