"""命令行入口：shadowvox 干声.wav 原曲.mp3 [选项]"""
import argparse
import json
import sys
from pathlib import Path

from . import AUTHOR, __version__
from .pipeline import Options, run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('desktop', 'batch', 'check-update'):
        return desktop_cli(argv)
    ap = argparse.ArgumentParser(
        prog="shadowvox",
        description="给一轨干声和原曲，自动按原唱音高修正并混成成品。",
        epilog="桌面处理链：shadowvox desktop --help。批量处理：shadowvox batch --help。更新：shadowvox check-update。",
    )
    ap.add_argument('--version', action='version', version=f'ShadowVox {__version__} · {AUTHOR}')
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
    ap.add_argument("--dry-has-backing", action="store_true", help="从含伴奏的干声录音中先分离演唱人声")
    ap.add_argument("--mute-reference-gaps", action="store_true", help="按对齐后的原唱唱段静音空档干声")
    ap.add_argument("--no-mix", action="store_true", help="只输出修后干声，不与伴奏混音")
    ap.add_argument("--mix-effects", action="store_true", help="开启轻度 EQ、压缩、混响、伴奏避让与响度平衡")
    ap.add_argument("--master-density", type=float, metavar="0..1",
                    help="使用 ZL 母带密度压缩；0 关闭总线压缩，与 --mix-effects 同用")
    ap.add_argument("--no-cache", action="store_true", help="不复用分离缓存")
    ap.add_argument("--work", help="工作目录（分离缓存）")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.master_density is not None and (not a.mix_effects or not 0. <= a.master_density <= 1.):
        ap.error('--master-density 需要 --mix-effects，范围为 0..1')

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
        master_density=a.master_density,
        denoise=not a.no_denoise,
        dry_has_backing=a.dry_has_backing,
        mute_reference_gaps=a.mute_reference_gaps,
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


def desktop_cli(argv):
    """Same inputs, perceptual controls and versioned output as the workbench."""
    from .jobs import DEFAULT_PARAMS, run_batch

    mode, *args = argv
    ap = argparse.ArgumentParser(prog=f'shadowvox {mode}')
    if mode == 'check-update':
        from .updates import check_update
        ap.parse_args(args)
        result = check_update()
        print(json.dumps(result, ensure_ascii=False))
        return int(result['state'] in ('error', 'limited', 'unavailable'))
    if mode == 'desktop':
        ap.add_argument('dry', help='演唱干音')
        ap.add_argument('song', help='含原唱和伴奏的原曲')
    else:
        ap.add_argument('dry', nargs='*', help='使用同一原曲的多个干音')
        ap.add_argument('--song', help='为多个干音指定同一首原曲')
        ap.add_argument('--manifest', type=Path, help='JSON 任务数组，每条包含 dry、song，可带 dry_has_backing')
    ap.add_argument('-o', '--output-dir', required=True, type=Path, help='版本化 WAV 和报告的输出目录')
    ap.add_argument('--work', help='工作目录（分离缓存）')
    ap.add_argument('--quiet', action='store_true', help='省略处理进度')
    ap.add_argument('--no-post', dest='mix_effects', action='store_false', help='关闭后期总开关')
    ap.add_argument('--dry-has-backing', action='store_true', help='先分离干音中混入的伴奏')
    for key, title in (('correction', '修正'), ('beauty', '美化'), ('fusion', '融合'),
                       ('ratio', '比例'), ('mixdown', '混音')):
        ap.add_argument('--'+key, type=int, default=DEFAULT_PARAMS[key], metavar='-100..100' if key == 'ratio' else '0..100',
                        help=f'{title}（默认 {DEFAULT_PARAMS[key]}）')
    a = ap.parse_args(args)
    for key in ('correction', 'beauty', 'fusion', 'ratio', 'mixdown'):
        if not (-100 if key == 'ratio' else 0) <= getattr(a, key) <= 100:
            ap.error(f'--{key} 超出范围')
    params = {key: getattr(a, key) for key in DEFAULT_PARAMS}
    if mode == 'desktop':
        items = [dict(dry=a.dry, song=a.song)]
    elif a.manifest:
        if a.song or a.dry:
            ap.error('--manifest 与 --song、干音参数互斥')
        try:
            items = json.loads(a.manifest.read_text(encoding='utf-8-sig'))
            if not isinstance(items, list) or not items:
                raise ValueError('需要非空 JSON 数组')
            for item in items:
                if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k] for k in ('dry', 'song')):
                    raise ValueError('每条任务需要 dry 和 song 路径')
                if 'dry_has_backing' in item and not isinstance(item['dry_has_backing'], bool):
                    raise ValueError('dry_has_backing 必须是布尔值')
                for key in ('dry', 'song'):
                    item[key] = str(a.manifest.resolve().parent/item[key])
        except (OSError, ValueError) as exc:
            ap.error(f'任务清单无效：{exc}')
    else:
        if not a.song or not a.dry:
            ap.error('请提供 --manifest，或指定 --song 和干音文件')
        items = [dict(dry=dry, song=a.song) for dry in a.dry]
    log = (lambda _: None) if a.quiet else (lambda message: print(message, flush=True))

    def show(result):
        if result['state'] == 'running':
            log(f'处理 {result["index"]+1}/{len(items)}')
        else:
            print(json.dumps(result, ensure_ascii=False), flush=True)

    try:
        results = run_batch(items, a.output_dir, params, work=a.work, log=log, update=show)
    except KeyboardInterrupt:
        return 130
    return int(any(result['state'] != 'done' for result in results))
