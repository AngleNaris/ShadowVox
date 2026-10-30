"""Compare Praat formant directions on the preferred moderate harmonic treatment."""

import json
import time

import numpy as np

import dsp_saturation_repair as saturation
import dsp_voice_comparisons as first
import renderer_comparisons


OUT = first.OUT / "v4_formants"
NOTES = {
    "A_selected": "中低频谐波适中，不经过 Praat",
    "B_neutral": "A + Praat 对照（共振峰不移动）",
    "C_down_light": "A + Praat（共振峰下移 3%）",
    "D_up_light": "A + Praat（共振峰上移 3%）",
    "E_down_strong": "A + Praat（共振峰下移 7%）",
    "F_up_strong": "A + Praat（共振峰上移 7%）",
}


def write_page(folder, song, summary, vocal_target, pitch):
    page = first.ORIGINAL_PAGE
    page = page.replace("渲染器对比 · 青花瓷", f"适中谐波与共振峰对比 · {song['title']}")
    page = page.replace("SHADOWVOX · 渲染器对比", f"SHADOWVOX · {song['title']}共振峰对比")
    page = page.replace(
        "青花瓷 · 同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。五版已统一到 __LUFS__ LUFS。",
        f"{song['title']} · 所有版本均使用你偏好的适中谐波。带伴奏已统一到 __LUFS__ LUFS；纯人声已统一到 {vocal_target:.1f} LUFS。")
    page = page.replace(
        "建议先比 A↔B（不修音高，只听渲染器本身的电音感），再比 C↔D↔E（修音后），最后 D↔F（节奏：只修可信入点 vs 整句贴齐）。",
        "先比 A↔B（Praat 本身），再比 B↔C↔D（上下 3%），最后比 E↔F（上下 7%）。")
    note = "纯人声和带伴奏分别匹配响度。请关注共鸣、本人辨识度、咬字以及是否有爆音或电音。"
    times = [t for result in pitch.values() for t in result.get("octave_like_times_seconds", [])]
    if times:
        note += f" Praat 版本约 {min(times):.1f} 秒处有短暂音高估计差异，可能是检测误差或处理瑕疵，请留意。"
    page = page.replace("纯人声未做试听响度匹配（按参考人声电平输出），只用于听瑕疵；电音感在纯人声里最明显。", note)
    other_id, other_title = (("turan", "突然好想你") if song["id"] == "qinghua"
                             else ("qinghua", "青花瓷"))
    page = page.replace(
        '<section class="panel"><h2><span class="idx">01</span>',
        f'<p><a href="../{other_id}/index.html">切换到《{other_title}》</a> · '
        f'<a href="../../v3_saturation/{song["id"]}/index.html">饱和对照</a></p>'
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
        reference, _ = first.load_clip(song["reference"], start, span, sr)
        accompaniment, _ = first.load_clip(song["accompaniment"], start, span, sr)
        vocal, reference = vocal[0], reference.mean(axis=0)
        common = first.baseline_vocal(vocal, sr)
        baseline_mix, _ = first.mixing.enhance(vocal, accompaniment, sr)
        bed = baseline_mix - (common[None, :] + first.mixing.room(common, sr) +
                              first.mixing.delay(common, sr))
        crop = slice(round(preroll * sr), round((preroll + song["duration"]) * sr))
        _, eq_info = first.reference_eq(common[crop], reference[crop], sr)
        eq = common.copy()
        for center, gain in zip(eq_info["filter_centers_hz"], eq_info["filter_gains_db"]):
            eq = first.mixing.bell(eq, sr, center, gain, .85)
        target_rms = first.audio.rms(common[crop])
        eq *= target_rms / max(first.audio.rms(eq[crop]), 1e-9)
        selected = saturation.warm_harmonics(eq, sr, crop, .16)
        variants = {"A_selected": selected}
        for name, ratio in (("B_neutral", 1.), ("C_down_light", .97),
                            ("D_up_light", 1.03), ("E_down_strong", .93),
                            ("F_up_strong", 1.07)):
            changed = first.formants(selected, sr, ratio)
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
        pitch = {name: first.pitch_comparison(dry["A_selected"][0], dry[name][0], sr)
                 for name in NOTES if name != "A_selected"}
        summary = {name: {"note": note, "listen_lufs": mix_lufs}
                   for name, note in NOTES.items()}
        write_page(folder, song, summary, dry_lufs, pitch)
        result = {
            "song": song["title"], "start_seconds": song["start"],
            "duration_seconds": song["duration"], "sample_rate": sr,
            "inputs": {key: str(song[key]) for key in ("vocal", "reference", "accompaniment")},
            "reference_eq": eq_info, "mix_lufs": mix_lufs, "vocal_lufs": dry_lufs,
            "harmonic_strength": .16, "mix_levels": mix_stats, "vocal_levels": dry_stats,
            "praat_pitch_check": pitch, "seconds": round(time.perf_counter() - started, 2),
            "note": "适中谐波为用户偏好；共振峰方向及音高稳定性等待人工试听。",
        }
        (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"song": song["title"], "seconds": result["seconds"],
                          "pitch": pitch, "vocal_lufs": dry_lufs, "mix_lufs": mix_lufs},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
