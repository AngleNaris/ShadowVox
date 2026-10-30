"""Second listening round: stronger saturation and formants in both directions."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

import dsp_voice_comparisons as first
import renderer_comparisons


OUT = first.OUT / "v2"
NOTES = {
    "A_current": "基线：现有均衡和动态处理",
    "B_reference_eq": "A + 原唱参考自适应 EQ",
    "C_saturation_light": "B + 轻度饱和（上一轮力度）",
    "D_saturation_medium": "B + 中度饱和",
    "E_saturation_strong": "B + 强度饱和",
    "F_praat_neutral": "D + Praat 对照（共振峰不移动）",
    "G_praat_down": "D + Praat（共振峰下移 3%）",
    "H_praat_up": "D + Praat（共振峰上移 3%）",
    "I_praat_up_strong": "D + Praat（共振峰上移 7%）",
}


def saturation(vocal, sr, drive, blend):
    band = sosfilt(butter(2, [100, 3500], btype="bandpass", fs=sr, output="sos"), vocal)
    level = max(3 * first.audio.rms(band), .03)
    up = resample_poly(band, 4, 1)
    bias = .14
    curved = level * (np.tanh(drive * up / level + bias) - np.tanh(bias))
    curved /= drive * (1 - np.tanh(bias) ** 2)
    difference = resample_poly(curved - up, 1, 4)[:len(vocal)]
    return (vocal + blend * difference).astype(np.float32)


def write_page(folder, song, summary, vocal_target, pitch):
    page = first.ORIGINAL_PAGE
    page = page.replace("渲染器对比 · 青花瓷", f"人声质感第二轮 · {song['title']}")
    page = page.replace("SHADOWVOX · 渲染器对比", f"SHADOWVOX · {song['title']}第二轮试听")
    page = page.replace(
        "青花瓷 · 同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。五版已统一到 __LUFS__ LUFS。",
        f"{song['title']} · 同一演唱片段，带伴奏已统一到 __LUFS__ LUFS；纯人声已统一到 {vocal_target:.1f} LUFS。")
    page = page.replace(
        "建议先比 A↔B（不修音高，只听渲染器本身的电音感），再比 C↔D↔E（修音后），最后 D↔F（节奏：只修可信入点 vs 整句贴齐）。",
        "先比 B↔C↔D↔E（饱和力度），再比 F↔G↔H↔I（共振峰不动、下移、上移）。")
    page = page.replace("键盘：1–6 选版本", "键盘：1–9 选版本")
    note = "两个试听内容分别匹配响度。请关注厚度、本人辨识度、咬字，以及饱和是否过于失真。"
    times = [t for result in pitch.values() for t in result.get("octave_like_times_seconds", [])]
    if times:
        note += f" Praat 版本约 {min(times):.1f} 秒处有短暂音高估计差异，可能是检测误差或处理瑕疵，请留意。"
    page = page.replace("纯人声未做试听响度匹配（按参考人声电平输出），只用于听瑕疵；电音感在纯人声里最明显。", note)
    other_id, other_title = (("turan", "突然好想你") if song["id"] == "qinghua"
                             else ("qinghua", "青花瓷"))
    page = page.replace(
        '<section class="panel"><h2><span class="idx">01</span>',
        f'<p><a href="../{other_id}/index.html">切换到《{other_title}》</a> · '
        f'<a href="../../{song["id"]}/index.html">上一轮试听</a></p>'
        '<section class="panel"><h2><span class="idx">01</span>')
    renderer_comparisons.PAGE = page
    try:
        renderer_comparisons.write_page(folder, summary)
    finally:
        renderer_comparisons.PAGE = first.ORIGINAL_PAGE


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for song in first.SONGS:
        started = time.perf_counter()
        folder = OUT / song["id"]
        folder.mkdir(parents=True, exist_ok=True)
        preroll, postroll = 2, 1
        start = song["start"] - preroll
        span = song["duration"] + preroll + postroll
        vocal, sr = first.load_clip(song["vocal"], start, span)
        ref, _ = first.load_clip(song["reference"], start, span, sr)
        accompaniment, _ = first.load_clip(song["accompaniment"], start, span, sr)
        if vocal.shape[0] != 1 or accompaniment.shape[0] != 2:
            raise ValueError("Unexpected vocal or accompaniment channel count")
        vocal, ref = vocal[0], ref.mean(axis=0)
        common = first.baseline_vocal(vocal, sr)
        baseline_mix, _ = first.mixing.enhance(vocal, accompaniment, sr)
        bed = baseline_mix - (common[None, :] + first.mixing.room(common, sr) +
                              first.mixing.delay(common, sr))
        crop = slice(round(preroll * sr), round((preroll + song["duration"]) * sr))
        _, eq_info = first.reference_eq(common[crop], ref[crop], sr)
        eq = common.copy()
        for center, gain in zip(eq_info["filter_centers_hz"], eq_info["filter_gains_db"]):
            eq = first.mixing.bell(eq, sr, center, gain, .85)
        target_rms = first.audio.rms(common[crop])
        eq *= target_rms / max(first.audio.rms(eq[crop]), 1e-9)

        variants = {"A_current": common, "B_reference_eq": eq}
        for name, drive, blend in (("C_saturation_light", 1.8, .55),
                                   ("D_saturation_medium", 3., .9),
                                   ("E_saturation_strong", 4.5, 1.)):
            processed = saturation(eq, sr, drive, blend)
            processed *= target_rms / max(first.audio.rms(processed[crop]), 1e-9)
            variants[name] = processed
        medium = variants["D_saturation_medium"]
        for name, ratio in (("F_praat_neutral", 1.), ("G_praat_down", .97),
                            ("H_praat_up", 1.03), ("I_praat_up_strong", 1.07)):
            changed = first.formants(medium, sr, ratio)
            changed *= target_rms / max(first.audio.rms(changed[crop]), 1e-9)
            variants[name] = changed

        dry, mixes = {}, {}
        for name, version in variants.items():
            if not np.all(np.isfinite(version)):
                raise ValueError(f"Non-finite signal in {name}")
            dry[name] = version[crop][None, :]
            mixes[name] = (bed + version[None, :] + first.mixing.room(version, sr) +
                           first.mixing.delay(version, sr))[:, crop]
        dry_lufs, dry_stats = first.save_level_matched(dry, sr, folder, "vocal")
        mix_lufs, mix_stats = first.save_level_matched(mixes, sr, folder, "listen")
        pitch = {name: first.pitch_comparison(dry["D_saturation_medium"][0], dry[name][0], sr)
                 for name in ("F_praat_neutral", "G_praat_down", "H_praat_up", "I_praat_up_strong")}
        summary = {name: {"note": note, "listen_lufs": mix_lufs}
                   for name, note in NOTES.items()}
        write_page(folder, song, summary, dry_lufs, pitch)
        result = {
            "song": song["title"], "start_seconds": song["start"],
            "duration_seconds": song["duration"], "sample_rate": sr,
            "inputs": {key: str(song[key]) for key in ("vocal", "reference", "accompaniment")},
            "reference_eq": eq_info, "mix_lufs": mix_lufs, "vocal_lufs": dry_lufs,
            "saturation": {"light": {"drive": 1.8, "blend": .55},
                           "medium": {"drive": 3., "blend": .9},
                           "strong": {"drive": 4.5, "blend": 1.}},
            "mix_levels": mix_stats, "vocal_levels": dry_stats, "praat_pitch_check": pitch,
            "seconds": round(time.perf_counter() - started, 2),
            "note": "短片试听，现有混音基线未逐样本复现正式母带；等待人工听感评价。",
        }
        (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"song": song["title"], "seconds": result["seconds"],
                          "pitch": pitch, "vocal_lufs": dry_lufs, "mix_lufs": mix_lufs},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
