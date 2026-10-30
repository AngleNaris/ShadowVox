"""本地自然度消融实验：不改原文件；输出同时间线、等响度试听和可重跑报告。

用法：python tools/naturalness_lab.py out/qinghua.report.json
"""
import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox import audio, config as C, rhythm
from shadowvox.align import estimate_shift, apply_shift
from shadowvox.f0 import FcpeExtractor, cent
from shadowvox.natural import runs
from shadowvox.pipeline import _place
from shadowvox.vocoder import Vocoder
from shadowvox.rubberband import render as rubberband
from eval_quality import err_curve, on_reference_grid

SR = C.VOC_SR
LABELS = {
    "original": "A 原始干声",
    "copy": "B 仅神经重合成",
    "rhythm_only": "C 仅节奏拉伸",
    "rhythm_copy": "D 节奏拉伸＋神经重合成",
    "frame": "E 逐帧修音（关节奏）",
    "frame_rhythm": "F 逐帧修音（开节奏，现有流程）",
    "natural": "G 音符中心修音＋神经声码器",
    "rubberband": "H 同一音符中心曲线＋Rubber Band R3",
    "rubberband_zero": "I Rubber Band R3 零移调对照",
}


def integrated_lufs(path):
    p = subprocess.run([audio._ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path),
                        "-af", "loudnorm=I=-23:TP=-1:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, check=True)
    log = p.stderr.decode(errors="replace")
    block = log[log.rfind("{"):log.rfind("}")+1]
    return float(json.loads(block)["input_i"])


def choose_clips(f0, x):
    """用信号统计选三类候选片段；不宣称已听辨其演唱技法。"""
    n = min(len(f0), len(x) // 441)
    f0 = f0[:n]
    cents = cent(f0)
    energy = np.sqrt(np.mean(x[:n*441].reshape(n, 441)**2, axis=1))
    active = energy > max(float(np.max(energy)) * .015, .0001)
    candidates = []
    for start in range(100, max(101, n-800), 100):
        end = min(n, start+800)
        v = f0[start:end] > 0
        if v.mean() < .35:
            continue
        ds = np.abs(np.diff(cents[start:end]))
        pair = v[:-1] & v[1:]
        smooth = median_filter(cents[start:end], size=9)
        stable = v & (np.abs(np.gradient(smooth)) < 5)
        longest = max((b-a for a,b in runs(stable)), default=0)
        slide = float(np.sum(np.clip(ds[pair], 0, 70)))
        uv = float(np.sum(energy[start:end][~v & active[start:end]]))
        candidates.append((start, [longest, slide, uv]))
    picked = []
    for k, label in enumerate(("长音候选", "转音候选", "清浊交替候选")):
        ranked = sorted(candidates, key=lambda c: c[1][k], reverse=True)
        available = [c for c in ranked if all(abs(c[0]-p["start_frame"]) >= 800 for p in picked)]
        if not available:
            available = ranked
        if not available:
            raise ValueError("素材太短或无足够浊音，无法自动选片")
        start = available[0][0]
        picked.append({"label": label, "start_frame": start,
                       "start_seconds": start/100, "end_seconds": (start+800)/100})
    return picked


def make_listening(out, tracks, f0):
    clips = choose_clips(f0, tracks["original"])
    rows, summary = [], []
    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)
    for i, clip in enumerate(clips, 1):
        start, end = int(clip["start_seconds"]*SR), int(clip["end_seconds"]*SR)
        cuts = {key: x[start:end].copy() for key, x in tracks.items()}
        levels = {}
        for key, cut in cuts.items():
            path = clips_dir / f"{i}_{key}.wav"
            sf.write(path, cut, SR, subtype="FLOAT")
            levels[key] = integrated_lufs(path)
            if not np.isfinite(levels[key]):
                raise ValueError(f"试听片段响度不可测: {path}")
        gains = {k: 10**((-23-levels[k])/20) for k in cuts}
        peak = max(float(np.max(np.abs(cuts[k])))*gains[k] for k in cuts)
        attenuation = min(1., 10**(-1/20)/max(peak, 1e-9))
        for key in cuts:
            sf.write(clips_dir / f"{i}_{key}.wav", cuts[key]*gains[key]*attenuation,
                     SR, subtype="PCM_24")
        clip["matched_lufs"] = -23 + 20*np.log10(attenuation)
        clip["original_lufs"] = levels
        rows.append(f'<h2>{i}. {clip["label"]} · {clip["start_seconds"]:.0f}–{clip["end_seconds"]:.0f} 秒</h2>')
        for key, label in LABELS.items():
            rows.append(f'<div class="track"><span>{html.escape(label)}</span><audio controls preload="none" src="clips/{i}_{key}.wav"></audio></div>')
        summary.append(clip)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>修音自然度试听</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f5f3ef;color:#252d30}h1{font-size:30px}h2{margin-top:36px}.track{display:flex;gap:20px;align-items:center;justify-content:space-between;padding:12px;background:white;border-bottom:1px solid #ddd}audio{width:360px;max-width:100%}@media(max-width:650px){.track{display:block}}</style>
<h1>修音自然度试听 · 第一轮</h1>
<p>每组为同一时间段，按综合响度匹配，再统一降低增益避免削波；无压缩、混响或伴奏遮盖。片段类别由信号统计选择，等待人工确认。</p>
<p><b>建议顺序：</b>A↔B 判断重合成损伤；A↔C 判断节奏拉伸；B↔E 判断逐帧修正；E↔G 判断保留表情；G↔H 比较引擎。F 是现有完整流程的重跑对照。I 检查传统引擎自身的零移调损伤。</p>
<p>关注金属感、颤音、滑音、字头和齿音。请记录“素材／片段编号／偏好的字母／听到的问题”。音准分数不能代替自然度试听。</p>
''' + "\n".join(rows) + '''<script>document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''
    (out / "listen.html").write_text(page, encoding="utf-8")
    return summary


def run(report_path, output=None):
    report_path = Path(report_path)
    old = json.loads(report_path.read_text(encoding="utf-8"))
    out = Path(output) if output else ROOT / "out" / "naturalness" / report_path.stem.split(".")[0]
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work" / Path(old["song"]).stem
    refs = list(work.glob("vocals_*.wav"))
    if len(refs) != 1:
        raise ValueError("需要唯一已分离参考轨")
    print(f"读取素材 {report_path.name}", flush=True)
    dry = audio.load_mono(old["dry"], SR)
    ref = audio.load_mono(refs[0], SR)
    ex, vocoder = FcpeExtractor(), Vocoder()
    df, rf = ex.extract(dry, SR), ex.extract(ref, SR)
    warped, rhythm_stats = rhythm.align_to_reference(dry, df, rf, SR)
    wf = ex.extract(warped, SR)
    offset = rhythm_stats.get("onset_offset_before_seconds", rhythm.onset_offset(df, rf))
    after_offset = rhythm.onset_offset(wf, rf)
    shifts = {}
    for key, source, off, mode in (("frame", df, offset, "frame"),
                                   ("natural", df, offset, "natural"),
                                   ("frame_rhythm", wf, after_offset, "frame")):
        shifts[key] = estimate_shift(source, rf, strength=.85, offset_override=off, pitch_mode=mode)
    n = len(ref)
    tracks = {"original": _place(dry, -offset, n, SR),
              "rhythm_only": _place(warped, -after_offset, n, SR)}
    for key, source, source_f0, off in (("copy", dry, df, offset),
                                       ("rhythm_copy", warped, wf, after_offset),
                                       ("frame", dry, df, offset),
                                       ("frame_rhythm", warped, wf, after_offset),
                                       ("natural", dry, df, offset)):
        print(f"生成 {LABELS[key]}", flush=True)
        target = apply_shift(source_f0, shifts[key].shift_cents) if key in shifts else source_f0
        tracks[key] = _place(vocoder.synth(source, target), -off, n, SR)
    rb_lengths = {}
    for key, shift in (("rubberband", shifts["natural"].shift_cents),
                       ("rubberband_zero", np.zeros_like(df))):
        print(f"生成 {LABELS[key]}", flush=True)
        rendered, rb_lengths[key] = rubberband(dry, shift, SR)
        tracks[key] = _place(rendered, -offset, n, SR)
    print("测量所有输出；保存全长音频", flush=True)
    f0s, errors = {}, {}
    for key in LABELS:
        sf.write(out / f"{key}.wav", tracks[key], SR, subtype="FLOAT")
        f0s[key] = ex.extract(tracks[key], SR)
        errors[key] = err_curve(on_reference_grid(f0s[key], 0, len(rf)), rf, 0)
    common = np.logical_and.reduce([np.isfinite(e) for e in errors.values()])
    metrics = {key: {"common_median_cents": float(np.median(e[common])),
                     "valid_frames": int(np.isfinite(e).sum())} for key,e in errors.items()}
    np.savez_compressed(out / "curves.npz", dry_f0=df, ref_f0=rf, **{k:r.shift_cents for k,r in shifts.items()})
    print("选择片段并匹配综合响度", flush=True)
    clips = make_listening(out, tracks, f0s["original"])
    result = {"source_report": str(report_path.resolve()), "dry": old["dry"],
              "reference": str(refs[0]), "offset_seconds": offset,
              "after_rhythm_offset_seconds": after_offset, "rhythm": rhythm_stats,
              "labels": LABELS, "common_frames": int(common.sum()), "metrics": metrics,
              "rubberband_length_delta_samples": rb_lengths, "clips": clips,
              "rubberband_engine": "official CLI 4.0.0 R3, formants preserved, pitchmap",
              "settings": {"strength": .85, "smooth_ms": 0, "octave": "auto"},
              "note": "全组共同浊音网格；音级误差忽略八度。原声和无节奏模式共用变时前入点偏移；节奏组用变时后偏移。未经人工盲听，不判定音色优劣。"}
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "common_frames": result["common_frames"], "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--out")
    args = parser.parse_args()
    run(args.report, args.out)
