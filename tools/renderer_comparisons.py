"""同一修音曲线、不同渲染器的整曲对比：神经声码器 / TD-PSOLA / Rubber Band R3。

用法：python -X utf8 tools/renderer_comparisons.py 干声 原曲 [输出目录]
0% 版本只做节奏与渲染，不修音高，用于单独判断渲染器本身的音色损伤。
"""
import json
import sys
import soundfile as sf
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shadowvox.pipeline import run, Options
from shadowvox.mixing import measure_loudness

VERSIONS = [("A_vocoder_0", "vocoder", 0., "guarded"), ("B_psola_0", "psola", 0., "guarded"),
            ("C_vocoder_75", "vocoder", .75, "guarded"), ("D_psola_75", "psola", .75, "guarded"),
            ("E_rubberband_75", "rubberband", .75, "guarded"), ("F_psola_full_75", "psola", .75, "full")]
NOTES = {"A_vocoder_0": "不修音高，只经神经声码器（现有引擎的音色损伤）",
         "B_psola_0": "不修音高，只经 PSOLA（新引擎的音色损伤）",
         "C_vocoder_75": "75% 修音，神经声码器（现有 GUI 默认）",
         "D_psola_75": "75% 修音，TD-PSOLA（推荐候选）",
         "E_rubberband_75": "75% 修音，Rubber Band R3 保共振峰",
         "F_psola_full_75": "75% 修音，PSOLA + 整句贴齐原唱（与 D 只差节奏方式）"}


def main(dry, song, out=ROOT / "out/renderer_compare"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "work" / Path(song).stem
    summary = {}
    for name, renderer, strength, rhythm_mode in VERSIONS:
        start = time.time()
        report = run(dry, song, out / f"{name}.wav", work_dir=work, log=lambda m: None,
                     opts=Options(strength=strength, pitch_mode="register", max_shift_cents=450,
                                  mix_effects=True, renderer=renderer, rhythm_mode=rhythm_mode))
        lufs, peak = measure_loudness(out / f"{name}.wav")
        summary[name] = dict(note=NOTES[name], renderer=renderer, strength=strength, rhythm_mode=rhythm_mode,
                             seconds=round(time.time() - start, 1), lufs=round(lufs, 2),
                             true_peak_dbtp=round(peak, 2),
                             mean_abs_shift_cents=round(report["align"]["mean_abs_shift_cents"], 1))
        print(name, summary[name], flush=True)
    # 试听版统一到共同安全响度，避免“更响 = 更好”的偏差；成品本身不改。
    target = min(-19., *(v["lufs"] - 1.5 - v["true_peak_dbtp"] for v in summary.values()))
    for name, item in summary.items():
        data, sr = sf.read(out / f"{name}.wav", dtype="float32")
        sf.write(out / f"{name}_listen.wav", data * 10 ** ((target - item["lufs"]) / 20), sr, subtype="PCM_24")
        item["listen_lufs"] = target
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。",
             f"试听请用 <名称>_listen.wav：已统一到 {target:.1f} LUFS。",
             "<名称>.wav 为原始成品（波形类渲染器峰值较高，母带后约轻 2dB），<名称>_vocal.wav 为纯人声。", ""]
    lines += [f"{k}：{v['note']}" for k, v in summary.items()]
    lines += ["", "建议先听 A 对 B（渲染器本身的电音感），再听 C / D / E（修音后），最后 D 对 F（节奏方式）。"]
    (out / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    write_page(out, summary)


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>渲染器对比 · 青花瓷</title>
<style>
*{box-sizing:border-box;border-radius:0}
body{margin:0;background:#fffafb;color:#261e2a;font:16px/1.6 "Microsoft YaHei UI",system-ui}
main{max-width:920px;margin:0 auto;padding:32px 24px}
h1{font:700 24px/1.3 Arial,"Microsoft YaHei UI";letter-spacing:2px;margin:0 0 4px}
.sub{color:#655563;margin:0 0 24px}
.panel{background:#fff1f6;border:1px solid #ecd1dc;padding:16px;margin:0 0 16px}
.idx{color:#b52f68;font:700 14px Consolas,monospace;margin-right:8px}
h2{font-size:15px;margin:0 0 12px;letter-spacing:1px}
.versions{display:grid;gap:8px}
.version{display:grid;grid-template-columns:44px 1fr;align-items:center;text-align:left;width:100%;
 background:#fff;border:1px solid #9b7184;padding:10px 12px;font:inherit;color:inherit;cursor:pointer;
 transition:background .15s,color .15s,border-color .15s}
.version:hover{background:#f9e0eb}
.version[aria-pressed=true]{background:#b52f68;border-color:#b52f68;color:#fff}
.version:focus-visible{outline:0;background:#f3cada;color:#261e2a;text-decoration:underline}
.version[aria-pressed=true]:focus-visible{background:#8f204f;color:#fff}
.key{font:700 15px Consolas,monospace}
.version small{display:block;opacity:.8;font-size:13px}
.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
button.ctl{background:#fff;border:1px solid #9b7184;font:inherit;padding:8px 18px;cursor:pointer;color:#261e2a;transition:background .15s}
button.ctl:hover{background:#f9e0eb}button.ctl:focus-visible{outline:0;background:#f3cada;text-decoration:underline}
button.primary{background:#b52f68;border-color:#b52f68;color:#fff;min-width:96px}
button.primary:hover{background:#9e2358}button.primary:focus-visible{background:#8f204f;color:#fff}
.seg{display:inline-flex;border:1px solid #9b7184}
.seg button{border:0;background:#fff;font:inherit;padding:8px 16px;cursor:pointer;color:#261e2a;transition:background .15s,color .15s}
.seg button[aria-pressed=true]{background:#b52f68;color:#fff}
.seg button:focus-visible{outline:0;text-decoration:underline}
.time{font:15px Consolas,monospace;color:#655563;min-width:120px}
input[type=range]{flex:1;min-width:200px;accent-color:#b52f68}
.hint{color:#655563;font-size:14px;margin:8px 0 0}
.note{color:#655563;font-size:14px}
</style><main>
<h1>SHADOWVOX · 渲染器对比</h1>
<p class="sub">青花瓷 · 同一干声、同一修音曲线、同样的节奏与混音，只更换渲染器。五版已统一到 __LUFS__ LUFS。</p>
<section class="panel"><h2><span class="idx">01</span>选择版本</h2><div class="versions">__BUTTONS__</div>
<p class="hint">建议先比 A↔B（不修音高，只听渲染器本身的电音感），再比 C↔D↔E（修音后），最后 D↔F（节奏：只修可信入点 vs 整句贴齐）。</p></section>
<section class="panel"><h2><span class="idx">02</span>播放</h2>
<div class="row"><button class="ctl primary" id="play">播放</button><span class="time" id="time">00:00 / 00:00</span>
<input id="seek" type="range" min="0" max="1000" value="0" aria-label="播放进度">
<div class="seg" role="group" aria-label="试听内容"><button id="mix" aria-pressed="true">带伴奏</button><button id="vocal" aria-pressed="false">纯人声</button></div></div>
<p class="hint">切换版本或伴奏时保持播放位置。键盘：1–6 选版本 · 空格 播放/暂停 · V 切换伴奏/纯人声 · ←/→ 跳 5 秒。</p>
<p class="note">纯人声未做试听响度匹配（按参考人声电平输出），只用于听瑕疵；电音感在纯人声里最明显。</p></section>
</main><script>
const V=__VERSIONS__;let cur=0,vocal=false,playing=false;const audios={};
function el(n,k){const key=n+(k?'_vocal':'_listen');if(!audios[key]){const a=new Audio(key+'.wav');a.preload='metadata';audios[key]=a;a.onended=()=>{playing=false;ui()}}return audios[key]}
function active(){return el(V[cur][0],vocal)}
function clock(s){s=Math.floor(s||0);return String(s/60|0).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function ui(){document.querySelectorAll('.version').forEach((b,i)=>b.setAttribute('aria-pressed',i===cur));
 mix.setAttribute('aria-pressed',!vocal);document.getElementById('vocal').setAttribute('aria-pressed',vocal);play.textContent=playing?'暂停':'播放'}
function swap(fn){const old=active(),t=old.currentTime;old.pause();fn();const a=active();
 const go=()=>{a.currentTime=Math.min(t,a.duration||t);if(playing)a.play()};a.readyState>=1?go():a.addEventListener('loadedmetadata',go,{once:true});ui()}
function select(i){if(i!==cur)swap(()=>cur=i)}
function toggle(){const a=active();if(playing){a.pause();playing=false}else{a.play();playing=true}ui()}
document.querySelectorAll('.version').forEach((b,i)=>b.onclick=()=>select(i));
play.onclick=toggle;mix.onclick=()=>vocal&&swap(()=>vocal=false);document.getElementById('vocal').onclick=()=>!vocal&&swap(()=>vocal=true);
seek.oninput=()=>{const a=active();if(a.duration)a.currentTime=seek.value/1000*a.duration};
setInterval(()=>{const a=active();time.textContent=clock(a.currentTime)+' / '+clock(a.duration);if(a.duration&&document.activeElement!==seek)seek.value=a.currentTime/a.duration*1000},200);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.key!==' ')return;
 if(e.key>='1'&&e.key<=String(V.length)){select(+e.key-1)}else if(e.key===' '){e.preventDefault();toggle()}
 else if(e.key.toLowerCase()==='v'){swap(()=>vocal=!vocal)}else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){const a=active();a.currentTime=Math.max(0,a.currentTime+(e.key==='ArrowRight'?5:-5))}});
ui();
</script></html>"""


def write_page(out, summary):
    import html
    letters = [name.split("_")[0] for name in summary]
    buttons = "".join(f'<button class="version" aria-pressed="false"><span class="key">{letter}</span>'
                      f'<span>{html.escape(item["note"])}<small>按 {i+1}</small></span></button>'
                      for i, (letter, item) in enumerate(zip(letters, summary.values())))
    page = (PAGE.replace("__BUTTONS__", buttons)
            .replace("__VERSIONS__", json.dumps([[name] for name in summary]))
            .replace("__LUFS__", f'{next(iter(summary.values()))["listen_lufs"]:.1f}'))
    (Path(out) / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--page"]:
        folder = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out/renderer_compare"
        write_page(folder, json.loads((folder / "summary.json").read_text(encoding="utf-8")))
    else:
        main(*sys.argv[1:])
