"""Generate isolated, loudness-matched DSP vocal-quality listening comparisons."""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from scipy.signal import butter, freqz, resample_poly, sosfilt, stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox import audio, mixing
import renderer_comparisons


OUT = ROOT / "out/dsp_voice_preview"
SONGS = [
    {
        "id": "qinghua", "title": "青花瓷", "start": 60, "duration": 20,
        "vocal": ROOT / "out/renderer_compare/F_psola_full_75_vocal.wav",
        "reference": ROOT / "work/周杰伦-青花瓷/vocals_e02ea0cd.wav",
        "accompaniment": ROOT / "work/周杰伦-青花瓷/inst_e02ea0cd.wav",
    },
    {
        "id": "turan", "title": "突然好想你", "start": 90, "duration": 20,
        "vocal": ROOT / "out/review/turan/修复后_vocal.wav",
        "reference": ROOT / "work/review_turan/vocals_da50270f.wav",
        "accompaniment": ROOT / "work/review_turan/inst_da50270f.wav",
    },
]
NOTES = {
    "A_current": "基线：现有均衡、动态和空间处理",
    "B_reference_eq": "A + 原唱参考自适应 EQ",
    "C_saturation": "B + 轻度谐波饱和",
    "D_praat_neutral": "C + Praat 对照（共振峰不移动）",
    "E_praat_light": "C + Praat（共振峰下移 3%）",
    "F_praat_strong": "C + Praat（共振峰下移 7%）",
}
BANDS = [(90, 180), (180, 350), (350, 700), (700, 1400),
         (1400, 2800), (2800, 5600), (5600, 10000)]


def load_clip(path, start, duration, expected_sr=None):
    info = sf.info(path)
    if expected_sr is not None and info.samplerate != expected_sr:
        raise ValueError(f"Sample-rate mismatch: {path}")
    samples, sr = sf.read(path, start=round(start * info.samplerate),
                          frames=round(duration * info.samplerate),
                          dtype="float32", always_2d=True)
    if len(samples) != round(duration * sr):
        raise ValueError(f"Not enough samples: {path}")
    return samples.T, sr


def baseline_vocal(vocal, sr):
    """The vocal portion of the current mixing.enhance processing."""
    y = sosfilt(butter(2, 75, btype="highpass", fs=sr, output="sos"), vocal)
    y = mixing.bell(y, sr, 350, -2., 1.)
    y, _ = mixing.compress(y, sr)
    y, _ = mixing.deess(y, sr)
    for freq, gain, q in ((180, 1.5, .8), (2800, 1.2, .8)):
        y = mixing.bell(y, sr, freq, gain, q)
    return audio.match_rms(y, audio.rms(vocal), max_gain_db=6).astype(np.float32)


def reference_eq(vocal, reference, sr):
    """Compare common active frames, then apply smooth, bounded broad-band EQ."""
    f, _, v = stft(vocal, sr, nperseg=4096, noverlap=3072, boundary=None)
    _, _, r = stft(reference, sr, nperseg=4096, noverlap=3072, boundary=None)
    v_energy = np.sum(np.abs(v) ** 2, axis=0)
    r_energy = np.sum(np.abs(r) ** 2, axis=0)
    active = ((v_energy > np.quantile(v_energy, .35)) &
              (r_energy > np.quantile(r_energy, .35)))
    if active.sum() < 10:
        raise ValueError("Not enough common active frames for reference EQ")
    vp = np.mean(np.abs(v[:, active]) ** 2, axis=1)
    rp = np.mean(np.abs(r[:, active]) ** 2, axis=1)
    anchor = (f >= 700) & (f < 2800)
    anchor_diff = 10 * np.log10((rp[anchor].sum() + 1e-15) / (vp[anchor].sum() + 1e-15))
    diffs = []
    for low, high in BANDS:
        band = (f >= low) & (f < high)
        diffs.append(10 * np.log10((rp[band].sum() + 1e-15) /
                                    (vp[band].sum() + 1e-15)) - anchor_diff)
    gains = np.clip(np.asarray(diffs) * .28, -1.2, 1.8)
    gains[0:2] = np.maximum(gains[0:2], -.8)
    gains[-1] = np.clip(gains[-1], -.5, .5)
    centers = np.sqrt(np.prod(np.asarray(BANDS), axis=1))

    def magnitude_db(values):
        response = np.ones(8192, dtype=np.complex128)
        for center, gain in zip(centers, values):
            amp = 10 ** (gain / 40)
            w = 2 * np.pi * center / sr
            alpha = np.sin(w) / (2 * .85)
            b = [1 + alpha * amp, -2 * np.cos(w), 1 - alpha * amp]
            a = [1 + alpha / amp, -2 * np.cos(w), 1 - alpha / amp]
            _, h = freqz(b, a, worN=8192)
            response *= h
        return 20 * np.log10(np.maximum(np.abs(response), 1e-12))

    max_db = float(np.max(np.abs(magnitude_db(gains))))
    if max_db > 2.8:
        gains *= 2.8 / max_db
    output = vocal.copy()
    for center, gain in zip(centers, gains):
        output = mixing.bell(output, sr, center, float(gain), .85)
    return output, {
        "common_active_frames": int(active.sum()),
        "band_differences_db": np.round(diffs, 2).tolist(),
        "filter_centers_hz": np.round(centers, 1).tolist(),
        "filter_gains_db": np.round(gains, 2).tolist(),
        "max_abs_response_db": round(float(np.max(np.abs(magnitude_db(gains)))), 2),
    }


def saturate(vocal, sr):
    """Blend a mildly saturated midrange residual; oversample the nonlinearity."""
    band = sosfilt(butter(2, [100, 3500], btype="bandpass", fs=sr, output="sos"), vocal)
    level = max(3 * audio.rms(band), .03)
    up = resample_poly(band, 4, 1)
    drive, bias = 1.8, .14
    curved = level * (np.tanh(drive * up / level + bias) - np.tanh(bias))
    curved /= drive * (1 - np.tanh(bias) ** 2)
    added = resample_poly(curved - up, 1, 4)[:len(vocal)]
    return (vocal + .55 * added).astype(np.float32)


def formants(vocal, sr, ratio):
    source = parselmouth.Sound(vocal.astype(np.float64), sampling_frequency=sr)
    out = call(source, "Change gender", 65, 1100, ratio, 0, 1, 1).values[0]
    if abs(len(out) - len(vocal)) > 2:
        raise ValueError(f"Praat changed the duration: {len(vocal)} -> {len(out)}")
    return np.pad(out, (0, max(0, len(vocal) - len(out))))[:len(vocal)].astype(np.float32)


def pitch_comparison(source, processed, sr):
    def f0(x):
        return parselmouth.Sound(x.astype(np.float64), sr).to_pitch(
            time_step=.01, pitch_floor=65, pitch_ceiling=1100).selected_array["frequency"]
    a, b = f0(source), f0(processed)
    n = min(len(a), len(b))
    valid = (a[:n] > 0) & (b[:n] > 0)
    cents = np.abs(1200 * np.log2(b[:n][valid] / a[:n][valid]))
    if not len(cents):
        return {"common_voiced_frames": 0}
    octave_frames = np.flatnonzero(valid)[cents > 600]
    return {"common_voiced_frames": int(len(cents)),
            "median_abs_cents": round(float(np.median(cents)), 1),
            "p90_abs_cents": round(float(np.quantile(cents, .9)), 1),
            "octave_like_frames": int(len(octave_frames)),
            "octave_like_times_seconds": np.round(octave_frames * .01, 2).tolist()}


def save_level_matched(tracks, sr, folder, suffix):
    stats = {}
    with tempfile.TemporaryDirectory(prefix="shadowvox-dsp-listen-") as temp:
        for name, data in tracks.items():
            path = Path(temp) / f"{name}.wav"
            sf.write(path, data.T, sr, subtype="FLOAT")
            lufs, peak = mixing.measure_loudness(path)
            if not np.isfinite(lufs) or not np.isfinite(peak):
                raise ValueError(f"Invalid loudness or peak: {name}")
            stats[name] = {"input_lufs": lufs, "input_true_peak_dbtp": peak}
        target = min(-19., *(v["input_lufs"] - 1.7 - v["input_true_peak_dbtp"]
                             for v in stats.values()))
        for name, data in tracks.items():
            gain = target - stats[name]["input_lufs"]
            path = folder / f"{name}_{suffix}.wav"
            sf.write(path, (data * 10 ** (gain / 20)).T, sr, subtype="PCM_24")
            lufs, peak = mixing.measure_loudness(path)
            if abs(lufs - target) > .2 or peak > -1.5:
                raise ValueError(f"Output level validation failed: {path}, {lufs}, {peak}")
            stats[name].update(output_lufs=round(lufs, 2),
                               output_true_peak_dbtp=round(peak, 2),
                               gain_db=round(gain, 2))
    return round(target, 2), stats


def write_page(folder, song, summary, vocal_target, pitch):
    page = renderer_comparisons.PAGE
    page = page.replace("渲染器对比 · 青花瓷", f"人声质感对比 · {song['title']}")
    page = page.replace("SHADOWVOX · 渲染器对比", f"SHADOWVOX · {song['title']}人声质感对比")
    page = page.replace(
        "青花瓷 · 同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。五版已统一到 __LUFS__ LUFS。",
        f"{song['title']} · 同一演唱片段，带伴奏已统一到 __LUFS__ LUFS；纯人声已统一到 {vocal_target:.1f} LUFS。")
    page = page.replace(
        "建议先比 A↔B（不修音高，只听渲染器本身的电音感），再比 C↔D↔E（修音后），最后 D↔F（节奏：只修可信入点 vs 整句贴齐）。",
        "建议依次对比 A↔B（参考 EQ）、B↔C（饱和）、C↔D（Praat 本身）、D↔E↔F（共振峰调整）。")
    note = "两个试听内容分别匹配响度。请关注厚度、本人辨识度、咬字和是否有电音或失真。"
    problem_times = [t for result in pitch.values() for t in result.get("octave_like_times_seconds", [])]
    if problem_times:
        note += f" Praat 版本约 {min(problem_times):.1f} 秒处有短暂音高估计差异，可能是检测误差或处理瑕疵，请留意。"
    page = page.replace("纯人声未做试听响度匹配（按参考人声电平输出），只用于听瑕疵；电音感在纯人声里最明显。", note)
    other_id, other_title = (("turan", "突然好想你") if song["id"] == "qinghua"
                             else ("qinghua", "青花瓷"))
    page = page.replace("<section class=\"panel\"><h2><span class=\"idx\">01</span>",
                        f'<p><a href="../{other_id}/index.html">切换到《{other_title}》</a></p>'
                        '<section class="panel"><h2><span class="idx">01</span>')
    renderer_comparisons.PAGE = page
    try:
        renderer_comparisons.write_page(folder, summary)
    finally:
        renderer_comparisons.PAGE = ORIGINAL_PAGE


ORIGINAL_PAGE = renderer_comparisons.PAGE


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for song in SONGS:
        started = time.perf_counter()
        folder = OUT / song["id"]
        folder.mkdir(parents=True, exist_ok=True)
        preroll, postroll = 2, 1
        start = song["start"] - preroll
        span = song["duration"] + preroll + postroll
        vocal, sr = load_clip(song["vocal"], start, span)
        ref, _ = load_clip(song["reference"], start, span, sr)
        accompaniment, _ = load_clip(song["accompaniment"], start, span, sr)
        if vocal.shape[0] != 1 or accompaniment.shape[0] != 2:
            raise ValueError("Unexpected vocal or accompaniment channel count")
        vocal, ref = vocal[0], ref.mean(axis=0)
        common = baseline_vocal(vocal, sr)
        baseline_mix, _ = mixing.enhance(vocal, accompaniment, sr)
        bed = baseline_mix - (common[None, :] + mixing.room(common, sr) + mixing.delay(common, sr))
        crop = slice(round(preroll * sr), round((preroll + song["duration"]) * sr))
        _, eq_info = reference_eq(common[crop], ref[crop], sr)
        # Reapply the same derived filters to pre-roll and post-roll to preserve continuity.
        eq = common.copy()
        for center, gain in zip(eq_info["filter_centers_hz"], eq_info["filter_gains_db"]):
            eq = mixing.bell(eq, sr, center, gain, .85)
        target_rms = audio.rms(common[crop])
        eq *= target_rms / max(audio.rms(eq[crop]), 1e-9)
        saturated = saturate(eq, sr)
        saturated *= target_rms / max(audio.rms(saturated[crop]), 1e-9)
        variants = {"A_current": common, "B_reference_eq": eq, "C_saturation": saturated}
        for name, ratio in (("D_praat_neutral", 1.), ("E_praat_light", .97), ("F_praat_strong", .93)):
            changed = formants(saturated, sr, ratio)
            changed *= target_rms / max(audio.rms(changed[crop]), 1e-9)
            variants[name] = changed
        dry, mixes = {}, {}
        for name, version in variants.items():
            if not np.all(np.isfinite(version)):
                raise ValueError(f"Non-finite signal in {name}")
            dry[name] = version[crop][None, :]
            mixes[name] = (bed + version[None, :] + mixing.room(version, sr) +
                           mixing.delay(version, sr))[:, crop]
        dry_lufs, dry_stats = save_level_matched(dry, sr, folder, "vocal")
        mix_lufs, mix_stats = save_level_matched(mixes, sr, folder, "listen")
        pitch = {name: pitch_comparison(dry["C_saturation"][0], dry[name][0], sr)
                 for name in ("D_praat_neutral", "E_praat_light", "F_praat_strong")}
        summary = {name: {"note": note, "listen_lufs": mix_lufs}
                   for name, note in NOTES.items()}
        write_page(folder, song, summary, dry_lufs, pitch)
        result = {
            "song": song["title"], "start_seconds": song["start"],
            "duration_seconds": song["duration"], "sample_rate": sr,
            "inputs": {key: str(song[key]) for key in ("vocal", "reference", "accompaniment")},
            "reference_eq": eq_info, "mix_lufs": mix_lufs, "vocal_lufs": dry_lufs,
            "mix_levels": mix_stats, "vocal_levels": dry_stats,
            "praat_pitch_check": pitch, "seconds": round(time.perf_counter() - started, 2),
            "note": "短片试听，现有混音基线未逐样本复现正式母带；等待人工听感评价。",
        }
        (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"song": song["title"], "seconds": result["seconds"],
                          "eq": eq_info, "pitch": pitch,
                          "vocal_lufs": dry_lufs, "mix_lufs": mix_lufs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
