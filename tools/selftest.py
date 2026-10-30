"""端到端自检：合成一首"已知标准答案"的歌，验证分离→整体对齐→修音→伴奏合并。

连续合成旋律只有一个浊音段，不能提供有限节奏算法要求的多个独立起音证据。
本测试关闭局部节奏变时以独立验收音高；节奏的自动整体延迟、局部修正与拒绝
行为由 test_desktop_audio.py 的多乐句/漏唱/超限及单起音用例验证。

用法: python tools/selftest.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shadowvox import audio, config as C            # noqa: E402
from shadowvox.f0 import FcpeExtractor              # noqa: E402
from eval_quality import compare               # noqa: E402

SR = C.VOC_SR
OUT = ROOT / "tools" / "_selftest"
OUT.mkdir(parents=True, exist_ok=True)


def harmonic_voice(f0_curve, sr=SR, seed=0):
    """用谐波叠加近似人声：1..18 次谐波，高频滚降 + 轻微噪声。"""
    rng = np.random.default_rng(seed)
    n = len(f0_curve)
    phase = 2 * np.pi * np.cumsum(f0_curve) / sr
    y = np.zeros(n, dtype=np.float64)
    for k in range(1, 19):
        y += (1.0 / k ** 1.1) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    y += 0.004 * rng.standard_normal(n)
    env = np.ones(n)
    fade = int(0.02 * sr)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (0.25 * y / np.abs(y).max() * env).astype(np.float32)


def make_melody(notes, sec_per_note=0.8, sr=SR):
    """notes: [(midi, vibrato_cents)] → (f0 曲线 @sr, 音符边界采样点)"""
    total = int(len(notes) * sec_per_note * sr)
    seg = int(sec_per_note * sr)
    f0 = np.zeros(total, dtype=np.float64)
    bounds = []
    for i, (midi, vib) in enumerate(notes):
        s, e = i * seg, min((i + 1) * seg, total)
        t = np.arange(e - s) / sr
        base = 440.0 * 2 ** ((midi - 69) / 12)
        glide = np.ones(e - s)
        ramp = int(0.04 * sr)
        if i > 0:
            prev = 440.0 * 2 ** ((notes[i - 1][0] - 69) / 12)
            glide[:ramp] = np.linspace(prev / base, 1.0, ramp)
        f0[s:e] = base * glide * 2 ** (vib * np.sin(2 * np.pi * 5.4 * t) / 1200)
        bounds.append((s, e))
    return f0, bounds


def detune(f0, bounds, rng, amount_cents=55):
    out = f0.copy()
    for s, e in bounds:
        out[s:e] *= 2 ** (rng.uniform(-amount_cents, amount_cents) / 1200)
    return out


def main():
    rng = np.random.default_rng(7)
    notes = [(60, 0), (62, 20), (64, 30), (65, 0), (67, 25), (64, 15),
             (62, 30), (60, 0), (67, 20), (65, 25)]
    ref_f0, bounds = make_melody(notes)

    ref_vocal = harmonic_voice(ref_f0, seed=1)
    # 伴奏：低频和弦铺底
    t = np.arange(len(ref_vocal)) / SR
    pad = sum(0.06 * np.sin(2 * np.pi * f * t) for f in (130.81, 164.81, 196.0, 261.63))
    pad = pad.astype(np.float32)
    song = np.vstack([ref_vocal + pad, ref_vocal + pad])
    song_path = OUT / "song.wav"
    audio.save(song_path, song, SR)

    # 干声：跑调 ±55 音分，整体晚 0.35s，另加底噪
    bad_f0 = detune(ref_f0, bounds, rng)
    bad = harmonic_voice(bad_f0, seed=2)
    delay = int(0.35 * SR)
    bad = np.concatenate([np.zeros(delay, np.float32), bad])
    bad += 0.002 * rng.standard_normal(len(bad)).astype(np.float32)
    dry_path = OUT / "dry.wav"
    audio.save(dry_path, bad, SR)

    out_path = OUT / "mixed.wav"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "run.py"), str(dry_path), str(song_path),
           "-o", str(out_path), "--no-cache", "--no-rhythm"]
    print("$", " ".join(cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return 1

    vocal_path = out_path.with_name(out_path.stem + "_vocal.wav")
    fixed = audio.load_mono(vocal_path, SR)

    # 评估：把干声/修后/原唱按已知对齐关系对齐后比较音高
    fc = FcpeExtractor()
    f_dry = fc.extract(bad, SR)
    f_fix = fc.extract(fixed, SR)
    f_ref = fc.extract(ref_vocal, SR)

    # 已知延迟是独立真值；不可把 44.1k 采样点除以 16k 的 hop=160。
    result = compare(f_dry, f_fix, f_ref, delay / SR)
    e_dry, e_fix = result["before_median"], result["after_median"]
    n1 = n2 = result["common_frames"]
    print("\n── 音高准确度（相对原唱，中位绝对值）──")
    print(f"  修音前 {e_dry:6.1f} 音分  (n={n1})")
    print(f"  修音后 {e_fix:6.1f} 音分  (n={n2})")
    rep = json.loads(out_path.with_suffix(".report.json").read_text(encoding="utf-8"))
    print(f"  报告: 偏移 {rep['align']['offset_seconds']:+.2f}s, "
          f"平均修正 {rep['align']['mean_abs_shift_cents']:.0f} 音分")
    offset_error = abs(rep['align']['offset_seconds'] - delay / SR)
    return 0 if e_fix < e_dry * 0.6 and offset_error < .05 else 2


if __name__ == "__main__":
    sys.exit(main())
