"""Compare the reported harsh saturation with low-frequency harmonic alternatives."""

import json
import time

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

import dsp_voice_comparisons as first
import dsp_voice_comparisons_v2 as second
import renderer_comparisons


OUT = first.OUT / "v3_saturation"
NOTES = {
    "A_eq_only": "参考 EQ，无额外饱和",
    "B_old_medium": "上一轮中度饱和",
    "C_old_strong": "上一轮强度饱和（反馈有爆音）",
    "D_warm_medium": "新方案：中低频谐波，适中",
    "E_warm_strong": "新方案：中低频谐波，更明显",
}


def warm_harmonics(vocal, sr, crop, strength):
    """Blend band-limited even harmonics, leaving the original high end intact."""
    band = sosfiltfilt(butter(2, [100, 1100], btype="bandpass", fs=sr, output="sos"), vocal)
    level = max(3 * first.audio.rms(band[crop]), .001)
    up = resample_poly(band, 4, 1) / level
    harmonics = resample_poly(np.tanh(1.3 * up) ** 2, 1, 4)[:len(vocal)]
    harmonics = sosfiltfilt(butter(3, [90, 2600], btype="bandpass", fs=sr, output="sos"), harmonics)
    target = first.audio.rms(vocal[crop])
    harmonics *= strength * target / max(first.audio.rms(harmonics[crop]), 1e-9)
    result = vocal + harmonics
    result *= target / max(first.audio.rms(result[crop]), 1e-9)
    return result.astype(np.float32)


def band_comparison(reference, signal, sr):
    frequencies, ref_power = welch(reference, sr, nperseg=4096)
    _, power = welch(signal, sr, nperseg=4096)
    result = []
    for low, high in ((100, 350), (350, 1500), (1500, 3500), (3500, 6000), (6000, 12000)):
        mask = (frequencies >= low) & (frequencies < high)
        change = 10 * np.log10((power[mask].sum() + 1e-15) /
                               (ref_power[mask].sum() + 1e-15))
        result.append(round(float(change), 2))
    return result


def write_page(folder, song, summary, vocal_target):
    page = first.ORIGINAL_PAGE
    page = page.replace("渲染器对比 · 青花瓷", f"饱和失真对比 · {song['title']}")
    page = page.replace("SHADOWVOX · 渲染器对比", f"SHADOWVOX · {song['title']}饱和对照")
    page = page.replace(
        "青花瓷 · 同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。五版已统一到 __LUFS__ LUFS。",
        f"{song['title']} · 只比较饱和，未做 Praat 共振峰调整。带伴奏已统一到 __LUFS__ LUFS；纯人声已统一到 {vocal_target:.1f} LUFS。")
    page = page.replace(
        "建议先比 A↔B（不修音高，只听渲染器本身的电音感），再比 C↔D↔E（修音后），最后 D↔F（节奏：只修可信入点 vs 整句贴齐）。",
        "先比 A↔B↔C（原方案），再比 A↔D↔E（新方案），重点听厚度是否增加，以及刺、裂或爆音是否减少。")
    page = page.replace("键盘：1–6 选版本", "键盘：1–5 选版本")
    page = page.replace("纯人声未做试听响度匹配（按参考人声电平输出），只用于听瑕疵；电音感在纯人声里最明显。",
                        "纯人声和带伴奏分别匹配响度。C 是用户反馈有爆音的旧强档；新方案仍需人工确认听感。")
    other_id, other_title = (("turan", "突然好想你") if song["id"] == "qinghua"
                             else ("qinghua", "青花瓷"))
    page = page.replace(
        '<section class="panel"><h2><span class="idx">01</span>',
        f'<p><a href="../{other_id}/index.html">切换到《{other_title}》</a> · '
        f'<a href="../../v2/{song["id"]}/index.html">上一轮试听</a></p>'
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
        variants = {"A_eq_only": eq}
        for name, drive, blend in (("B_old_medium", 3., .9), ("C_old_strong", 4.5, 1.)):
            version = second.saturation(eq, sr, drive, blend)
            version *= target_rms / max(first.audio.rms(version[crop]), 1e-9)
            variants[name] = version
        for name, strength in (("D_warm_medium", .16), ("E_warm_strong", .28)):
            variants[name] = warm_harmonics(eq, sr, crop, strength)
        dry, mixes = {}, {}
        for name, version in variants.items():
            if not np.all(np.isfinite(version)):
                raise ValueError(f"Non-finite signal in {name}")
            dry[name] = version[crop][None, :]
            mixes[name] = (bed + version[None, :] + first.mixing.room(version, sr) +
                           first.mixing.delay(version, sr))[:, crop]
        dry_lufs, dry_stats = first.save_level_matched(dry, sr, folder, "vocal")
        mix_lufs, mix_stats = first.save_level_matched(mixes, sr, folder, "listen")
        spectrum = {name: band_comparison(dry["A_eq_only"][0], value[0], sr)
                    for name, value in dry.items()}
        summary = {name: {"note": note, "listen_lufs": mix_lufs}
                   for name, note in NOTES.items()}
        write_page(folder, song, summary, dry_lufs)
        result = {
            "song": song["title"], "start_seconds": song["start"],
            "duration_seconds": song["duration"], "sample_rate": sr,
            "inputs": {key: str(song[key]) for key in ("vocal", "reference", "accompaniment")},
            "reference_eq": eq_info, "mix_lufs": mix_lufs, "vocal_lufs": dry_lufs,
            "bands_hz": [[100, 350], [350, 1500], [1500, 3500], [3500, 6000], [6000, 12000]],
            "band_changes_vs_eq_db": spectrum, "mix_levels": mix_stats, "vocal_levels": dry_stats,
            "seconds": round(time.perf_counter() - started, 2),
            "note": "新的中低频谐波方案需人工试听，尚不能断言无失真或更好听。",
        }
        (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"song": song["title"], "seconds": result["seconds"],
                          "bands": spectrum, "vocal_lufs": dry_lufs, "mix_lufs": mix_lufs},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
