"""命令行入口：shadowvox 干声.wav 原曲.mp3 [选项]"""
import argparse
import sys
from pathlib import Path

from . import config as C
from .pipeline import Options, run


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="shadowvox",
        description="傻瓜式修音：给一轨干声和原曲，自动按原唱音高修正并混成成品。",
    )
    ap.add_argument("dry", help="干声文件（wav/mp3/flac…）")
    ap.add_argument("song", help="原曲成品（含伴奏与人声）")
    ap.add_argument("-o", "--out", help="输出成品路径（默认 out/<干声名>_autotuned.wav）")
    ap.add_argument("--strength", type=float, default=0.85,
                    help="修正力度 0~1，默认 0.85")
    ap.add_argument("--max-shift", type=float, default=700.0,
                    help="单帧最大移调音分，默认 700（7 个半音）")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="frame 模式偏移平滑窗(ms)，0=逐帧跟随；不代表自然度最佳")
    ap.add_argument("--pitch-mode", choices=["frame", "natural", "register"], default="frame",
                    help="frame=原逐帧模式；natural=音符中心修正实验（忽略 --smooth）")
    ap.add_argument("--vocal-gain", type=float, default=0.0,
                    help="修后干声额外增益(dB)，默认 0（对齐原唱响度）")
    ap.add_argument("--octave", choices=["auto", "match"], default="auto",
                    help="八度策略：auto=干声在自己音区修正（默认）；"
                         "match=整体搬到原唱音区，适合低/高八度跟唱")
    ap.add_argument("--no-rhythm", action="store_true",
                    help="关闭入点级节奏对齐（默认开启：按原唱入点对齐，只修入点）")
    ap.add_argument("--rhythm-rate", type=float, default=0.10,
                    help="节奏对齐的局部时间伸缩上限，默认 0.10（10%%）。"
                         "调大跟得更紧但更容易听出拉伸，调小更自然但修得少")
    ap.add_argument("--sep-model", default="Kim_Vocal_2.onnx", help="人声分离模型")
    ap.add_argument("--no-denoise", action="store_true", help="关闭修音前的 AI 降噪")
    ap.add_argument("--no-mix", action="store_true", help="只输出修后干声，不与伴奏混音")
    ap.add_argument("--mix-effects", action="store_true", help="开启轻度 EQ、压缩、混响、伴奏避让与响度平衡")
    ap.add_argument("--no-cache", action="store_true", help="不复用分离缓存")
    ap.add_argument("--work", help="工作目录（分离缓存）")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    log = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    opts = Options(
        strength=a.strength,
        max_shift_cents=a.max_shift,
        smooth_ms=a.smooth,
        pitch_mode=a.pitch_mode,
        vocal_gain_db=a.vocal_gain,
        octave=a.octave,
        rhythm=not a.no_rhythm,
        rhythm_max_rate=a.rhythm_rate,
        separate_model=a.sep_model,
        reuse_stems=not a.no_cache,
        mix=not a.no_mix,
        mix_effects=a.mix_effects,
        denoise=not a.no_denoise,
    )
    try:
        report = run(a.dry, a.song, a.out, a.work, opts, log=log)
    except Exception as e:
        print(f"失败: {e}", file=sys.stderr)
        return 1

    if not a.quiet:
        al = report.get("align", {})
        print("\n── 修音报告 ──")
        print(f"  干声 {report['dry_seconds']}s / 原曲 {report['song_seconds']}s")
        print(f"  浊音占比 干声 {report['dry_voiced_ratio']:.1%} / 原唱 {report['ref_voiced_ratio']:.1%}")
        print(f"  整体偏移 {al.get('offset_seconds', 0):+.2f}s（已自动对齐）")
        rh = report.get("rhythm") or {}
        if rh.get("applied"):
            print(f"  节奏对齐 入点 {rh['onset_pairs']} 个，原入点偏差 "
                  f"P50={rh['onset_dev_p50_ms']:.0f}ms / P90={rh['onset_dev_p90_ms']:.0f}ms"
                  f"（限速 {rh['max_rate']:.0%}）")
        elif rh:
            print("  节奏对齐 未应用（入点太少或素材不适配）")
        if report.get("octave_shift"):
            print(f"  八度搬运 {report['octave_shift']:+.0f} 个八度（--octave match）")
        print(f"  平均修正 {al.get('mean_abs_shift_cents', 0):.0f} 音分，"
              f"峰值 {al.get('max_abs_shift_cents', 0):.0f} 音分")
        print(f"  对齐置信度 {al.get('mean_confidence', 0):.2f}")
        print(f"  成品 {report.get('output', report['vocal_stem'])}")
    return 0
