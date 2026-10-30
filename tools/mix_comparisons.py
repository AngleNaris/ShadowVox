"""把自然度实验的全长人声与同一伴奏合并，统一人声响度及总线静态增益。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox.audio import _ffmpeg_exe


def measure(path):
    result = subprocess.run([
        _ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path),
        "-af", "loudnorm=I=-23:TP=-1:LRA=11:print_format=json", "-f", "null", "-",
    ], capture_output=True, check=True)
    log = result.stderr.decode(errors="replace")
    data = json.loads(log[log.rfind("{"):log.rfind("}")+1])
    return {"lufs": float(data["input_i"]), "true_peak_dbtp": float(data["input_tp"])}


def measure_many(paths):
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(paths, pool.map(measure, paths)))


def make_page(out, labels, title, introduction=None):
    cards = []
    for key, label in labels.items():
        name = f"{label[0]}_{key}"
        cards.append(f'<section><h2>{html.escape(label)}</h2>'
                     f'<audio controls preload="metadata" src="{name}.mp3"></audio>'
                     f'<p><a href="{name}.wav" download>下载 WAV</a> · '
                     f'<a href="{name}.mp3" download>下载 MP3</a></p></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:17px/1.7 system-ui;max-width:950px;margin:32px auto;padding:0 20px;background:#f5f3ef;color:#252d30}h1{font-size:30px}h2{font-size:19px;margin:0 0 12px}section{background:white;padding:20px;margin:16px 0;border:1px solid #ddd;border-radius:8px}audio{width:100%}a{color:#087866}label{display:block;padding:12px 0}p{margin:8px 0}</style>
''' + f'<title>{html.escape(title)} · 完整混音对比</title><h1>{html.escape(title)} · 完整混音对比</h1>' + '''
<p>九个版本均为完整歌曲，使用同一条伴奏。人声按整曲综合响度匹配到参考人声，再用所有版本共同的静态增益预留峰值余量。未增加混响、压缩或美化。</p>
<p>优先比较 E / F / G / H；A 为未修音对照，B / C / D / I 用于定位音色或节奏处理的影响。伴奏来自原曲分离，可能残留原唱。</p>
<label><input id="sync" type="checkbox" checked> 切换版本时，接着上一版的位置播放</label>
<p><a href="../listen.html">返回短片段干声对比</a></p>
''' + "\n".join(cards) + '''
<script>
let last=null;
document.addEventListener('play', e=>{
 const next=e.target;if(next.tagName!=='AUDIO')return;
 if(last&&last!==next){const time=last.currentTime;last.pause();if(document.getElementById('sync').checked)next.currentTime=last.ended?0:Math.min(time,next.duration||time);}
 document.querySelectorAll('audio').forEach(a=>{if(a!==next)a.pause()});last=next;
},true);
</script></html>'''
    page = page.replace("九个版本", f"{len(labels)} 个版本")
    if introduction:
        page = page.replace("优先比较 E / F / G / H；A 为未修音对照，B / C / D / I 用于定位音色或节奏处理的影响。", html.escape(introduction))
    page = page.replace('<p><a href="../listen.html">返回短片段干声对比</a></p>',
                        '<p><a href="../listen.html">返回短片段干声对比</a></p>' if (out.parent / "listen.html").exists() else '')
    (out / "index.html").write_text(page, encoding="utf-8")


def run(folder, title):
    folder = Path(folder).resolve()
    lab = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    labels = lab["labels"]
    reference = Path(lab["reference"])
    accompaniment = reference.with_name(reference.name.replace("vocals_", "inst_", 1))
    inst, sr = sf.read(accompaniment, dtype="float32", always_2d=True)
    if inst.shape[1] != 2:
        raise ValueError("需要立体声伴奏")
    vocals = {key: folder / f"{key}.wav" for key in labels}
    out = folder / "full_mix"
    out.mkdir(exist_ok=True)
    print("测量整曲人声响度", flush=True)
    levels = measure_many([reference, *vocals.values()])
    target = levels[reference]["lufs"]
    gains = {k: target-levels[p]["lufs"] for k,p in vocals.items()}
    if not all(np.isfinite(list(gains.values()))):
        raise ValueError("人声响度不可测")
    with tempfile.TemporaryDirectory(prefix="_mix-", dir=out) as temp:
        work = Path(temp).resolve()
        if not work.is_relative_to(out.resolve()):
            raise ValueError("临时目录超出输出目录")
        raw = {}
        for key,path in vocals.items():
            vocal, vsr = sf.read(path, dtype="float32")
            if vsr != sr or vocal.ndim != 1 or len(vocal) != len(inst):
                raise ValueError(f"人声采样率、声道或长度不匹配: {path}")
            mixed = inst + vocal[:,None]*10**(gains[key]/20)
            if not np.isfinite(mixed).all():
                raise ValueError(f"混音存在非有限值: {key}")
            raw[key] = work / f"{key}.wav"
            sf.write(raw[key], mixed, sr, subtype="FLOAT")
        print("测量混音峰值，设置统一余量", flush=True)
        raw_levels = measure_many(list(raw.values()))
        bus_db = min(0., -2.-max(m["true_peak_dbtp"] for m in raw_levels.values()))
        outputs = {}
        for key,label in labels.items():
            name = f"{label[0]}_{key}"
            wav, mp3 = out / f"{name}.wav", out / f"{name}.mp3"
            data, _ = sf.read(raw[key], dtype="float32", always_2d=True)
            data *= 10**(bus_db/20)
            sf.write(wav, data, sr, subtype="PCM_24")
            subprocess.run([_ffmpeg_exe(), "-v", "error", "-y", "-i", str(wav),
                            "-c:a", "libmp3lame", "-b:a", "320k", str(mp3)], check=True)
            outputs[key] = {"label": label, "wav": wav.name, "mp3": mp3.name,
                            "vocal_gain_db": gains[key]}
            print(f"已导出 {label}", flush=True)
    print("校验 WAV / MP3 的响度、峰值与时长", flush=True)
    paths = [out / row[fmt] for row in outputs.values() for fmt in ("wav", "mp3")]
    final_levels = measure_many(paths)
    for row in outputs.values():
        for fmt in ("wav", "mp3"):
            path = out / row[fmt]
            info = sf.info(path)
            if info.samplerate != sr or info.channels != 2 or info.frames != len(inst):
                raise ValueError(f"导出格式/解码长度不符: {path}")
            row[fmt+"_measurement"] = final_levels[path]
            if final_levels[path]["true_peak_dbtp"] > -1.:
                raise ValueError(f"导出峰值余量不足: {path}")
    report = {"date": lab.get("date", "2026-10-26"), "title": title, "source": str(folder),
              "accompaniment": str(accompaniment), "sample_rate": sr,
              "frames": len(inst), "seconds": len(inst)/sr,
              "reference_vocal_lufs": target, "shared_bus_gain_db": bus_db,
              "method": "人声整曲综合响度匹配到参考，伴奏恒定；所有版本共用静态总线增益。保留局部动态差异。",
              "outputs": outputs}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_page(out, labels, title, lab.get("introduction"))
    listening = folder / "listen.html"
    page = listening.read_text(encoding="utf-8") if listening.exists() else ''
    link = f'<p><a href="full_mix/index.html"><b>试听完整歌曲（{len(labels)} 版，已合并伴奏）</b></a></p>'
    if page and 'href="full_mix/index.html"' not in page:
        page = page.replace('</h1>', '</h1>\n'+link, 1)
        listening.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(out), "seconds": report["seconds"],
                      "versions": len(outputs), "shared_bus_gain_db": bus_db}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    ap.add_argument("--title", default="青花瓷")
    args = ap.parse_args()
    run(args.folder, args.title)
