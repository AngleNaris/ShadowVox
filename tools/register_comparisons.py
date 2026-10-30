"""复用 D 的节奏波形，生成保留演唱者音区的三档修音及完整混音。"""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox import audio, rhythm
from shadowvox.align import estimate_shift, apply_shift
from shadowvox.f0 import FcpeExtractor, cent
from shadowvox.pipeline import _place
from shadowvox.vocoder import Vocoder
from eval_quality import err_curve, on_reference_grid
from mix_comparisons import run as mix


def write_readme(out):
    text = """青花瓷：D 基础上的三档音高修正
日期：2026-10-27

本目录所有音频都已合并同一条伴奏，完整时长 3 分 57.2 秒。
每个版本提供 WAV（44.1kHz、24bit、立体声）和 MP3（320kbps）。

D_rhythm_copy：节奏调整＋神经重合成，不主动修正音高。
F_frame_rhythm：原来的逐帧修音版本，用作旧方案对照。
J_light：轻度，修正比例 45%，保留较多原演唱的音高起伏。
K_medium：中度，修正比例 75%，建议从这个版本开始听。
L_strong：强度，修正比例 100%，音高变化收紧更明显，也更可能影响颤音和滑音。

八度原则
先把参考旋律匹配到演唱者原本所在的音区，再计算需要修正的音高差。
例如原唱 C4、演唱者在 C3 附近，就向 C3 修正，不强行拉到 C4。
按乐句判断八度关系；持续至少约 300ms 的音区变化可单独匹配。
短暂的八度跳变、残差大于 450 音分的位置保守跳过，清音不主动移调。
这是自动判断，不能保证所有真假声、气声或音高误检都识别正确。

力度是什么意思
三档使用相同节奏输入、八度保护、修正曲线算法和神经声码器，仅修正比例不同。
已核对节奏输入与原 D 完全一致；从节奏处理后的原人声重新合成，不对 D 成品重复处理。
先对音高差做约 50ms 的局部平滑，再按力度修正；边界另有短平滑过渡。
不再因为偏差达到 300 音分就把修正量自动降为零；但不确定位置仍会减弱或跳过。
100% 是可靠、经过平滑的音高差修正比例，不是保证每一帧完全唱准。
它也不是把整个人声变成原唱音色或强行改成原唱的八度。
强度增加可能让颤音变浅；听起来是否更自然，需要实际试听。

建议比较
先 D 对 K，再比较 J、K、L；用 F 判断与旧方案的区别。
注意长音、转音、句尾和真假声切换，分别判断音准与自然感。
请在这个目录内比较五版：整组重新统一人声响度及总线增益，
直接与旧目录播放可能有音量差，不能仅凭更响判断更好。
未额外增加混响、压缩或美化。伴奏来自原曲分离，可能残留原唱。

试听页：index.html
本机服务地址：http://127.0.0.1:8766/qinghua/register_strength/full_mix/index.html
默认切换版本时接着上一版的位置播放，可取消勾选。
report.json：本目录的文件清单、混音增益、响度和真峰值。
上一级 report.json：算法设置及音级误差；curves.npz：音高曲线。
音级误差忽略八度，不能用它单独证明音区正确或自然度更好。
"""
    folder = Path(out) / "full_mix"
    (folder / "README.txt").write_text(text, encoding="utf-8")
    page = (folder / "index.html").read_text(encoding="utf-8")
    link = '<p><a href="README.txt">纯文本说明</a> · <a href="../../full_mix/index.html">返回原九版混音</a></p>'
    if 'href="README.txt"' not in page:
        (folder / "index.html").write_text(page.replace('</h1>', '</h1>'+link, 1), encoding="utf-8")


def run():
    source = ROOT / "out/naturalness/qinghua"
    out = source / "register_strength"
    out.mkdir(exist_ok=True)
    old = json.loads((source / "report.json").read_text(encoding="utf-8"))
    sr = 44100
    labels = {"rhythm_copy": "D 节奏＋重合成（无修音对照）",
              "frame_rhythm": "F 旧版逐帧修音（对照）",
              "light": "J 轻度修音（45%，保留音区）",
              "medium": "K 中度修音（75%，保留音区）",
              "strong": "L 强度修音（100%，保留音区）"}
    print("读取原始素材、重新生成并核对 D 的节奏输入", flush=True)
    ex = FcpeExtractor()
    dry = audio.load_mono(old["dry"], sr)
    rf = np.load(source / "curves.npz")["ref_f0"]
    df = np.load(source / "curves.npz")["dry_f0"]
    warped, rhythm_stats = rhythm.align_to_reference(dry, df, rf, sr)
    wf = ex.extract(warped, sr)
    offset = old["after_rhythm_offset_seconds"]
    previous, _ = sf.read(source / "rhythm_only.wav", dtype="float32")
    placed = _place(warped, -offset, len(previous), sr)
    max_diff = float(np.max(np.abs(placed-previous)))
    if max_diff > 1e-6:
        raise ValueError(f"节奏输入与 D 不一致: {max_diff}")
    print(f"节奏输入与 D 一致，最大采样差 {max_diff}", flush=True)
    for key in ("rhythm_copy", "frame_rhythm"):
        shutil.copyfile(source / f"{key}.wav", out / f"{key}.wav")
    vocoder = Vocoder()
    curves = {"source_f0": wf, "reference_f0": rf}
    settings = {}
    for key, strength in (("light", .45), ("medium", .75), ("strong", 1.)):
        print(f"计算 {labels[key]}", flush=True)
        result = estimate_shift(wf, rf, strength=strength, max_shift_cents=450,
                                offset_override=offset, pitch_mode="register")
        shift = result.shift_cents
        assert np.isfinite(shift).all() and np.max(np.abs(shift)) <= 450*strength+1e-6
        target = apply_shift(wf, shift)
        assert np.all(target[wf == 0] == 0)
        curves[key+"_shift"] = shift
        curves[key+"_target_f0"] = target
        settings[key] = {"strength": strength, "max_shift_cents": 450, **result.stats}
        print(json.dumps(settings[key], ensure_ascii=False), flush=True)
        def progress(value):
            print(f"合成 {key}: {value:.0%}", flush=True)
        rendered = _place(vocoder.synth(warped, target, progress=progress), -offset, len(previous), sr)
        if not np.isfinite(rendered).all():
            raise ValueError("输出有非有限采样")
        sf.write(out / f"{key}.wav", rendered, sr, subtype="FLOAT")
    print("重新提取各版输出音高，测量共同浊音帧", flush=True)
    errors = {}
    for key in labels:
        y, _ = sf.read(out / f"{key}.wav", dtype="float32")
        f0 = ex.extract(y, sr)
        curves[key+"_output_f0"] = f0
        errors[key] = err_curve(on_reference_grid(f0, 0, len(rf)), rf, 0)
    common = np.logical_and.reduce([np.isfinite(e) for e in errors.values()])
    metrics = {k: {"common_median_cents": float(np.median(e[common])),
                    "common_mean_cents": float(np.mean(e[common]))} for k,e in errors.items()}
    np.savez_compressed(out / "curves.npz", **curves)
    report = {"date": "2026-10-27", "reference": old["reference"], "labels": labels,
              "source_report": str(source / "report.json"), "rhythm": rhythm_stats,
              "rhythm_input_max_sample_difference": max_diff,
              "after_rhythm_offset_seconds": offset, "settings": settings,
              "common_frames": int(common.sum()), "metrics": metrics,
              "introduction": "以 D 的节奏处理为基础比较 J 轻度、K 中度、L 强度；F 是旧修音对照。三档均按演唱者音区匹配八度，100% 表示可靠平滑音高差的修正比例，不是复制原唱八度。强档会更明显地收紧颤音。",
              "note": "音区匹配为启发式；450c 以上残差跳过；音级误差忽略八度，不能代替听感或独立证明音区正确。"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    mix(out, "青花瓷 · D 基础三档修音")
    write_readme(out)


if __name__ == "__main__":
    run()
