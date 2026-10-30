"""导出旧节奏、新保护节奏、轻混音三版等响度完整对照。"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shadowvox import audio
from mix_comparisons import measure, make_page


def run():
    folder=ROOT/'out/naturalness/qinghua/desktop_v1'
    folder.mkdir(exist_ok=True)
    gui=json.loads((ROOT/'out/desktop_preview/gui_evidence/result.json').read_text(encoding='utf-8'))
    mixed_path=Path(gui['output'])
    report=json.loads(mixed_path.with_suffix('.report.json').read_text(encoding='utf-8'))
    voice,sr=sf.read(report['vocal_stem'],dtype='float32')
    inst,_=sf.read(ROOT/'work/周杰伦-青花瓷/inst_e02ea0cd.wav',dtype='float32',always_2d=True)
    plain=inst+voice[:,None]
    old,_=sf.read(ROOT/'out/naturalness/qinghua/register_strength/full_mix/K_medium.wav',dtype='float32',always_2d=True)
    mixed,_=sf.read(mixed_path,dtype='float32',always_2d=True)
    tracks={'old':old,'guarded':plain,'mixed':mixed}
    labels={'old':'A 旧节奏 · 中档修音','guarded':'B 新节奏保护 · 中档修音 · 混音关闭',
            'mixed':'C 新节奏保护 · 中档修音 · 混音开启'}
    levels={}
    for key,data in tracks.items():
        path=folder/f'{labels[key][0]}_{key}.wav'
        sf.write(path,data,sr,subtype='FLOAT');levels[key]=measure(path)
    gains={key:-19-values['lufs'] for key,values in levels.items()}
    extra=min(0.,-1.5-max(levels[k]['true_peak_dbtp']+gains[k] for k in tracks))
    final={}
    for key,data in tracks.items():
        name=f'{labels[key][0]}_{key}'
        wav=folder/(name+'.wav');mp3=folder/(name+'.mp3')
        sf.write(wav,data*10**((gains[key]+extra)/20),sr,subtype='PCM_24')
        subprocess.run([audio._ffmpeg_exe(),'-v','error','-y','-i',str(wav),'-b:a','320k',str(mp3)],
                       check=True,**audio.subprocess_options())
        final[key]={'wav':measure(wav),'mp3':measure(mp3),'seconds':sf.info(wav).duration}
        assert sf.info(wav).frames==len(inst)
        assert final[key]['mp3']['true_peak_dbtp'] < -1
    make_page(folder,labels,'青花瓷 · 节奏保护与轻混音',
              '先 A/B 比较节奏保护，再 B/C 比较混音。B/C 使用同一修后人声，均为 75% 修音力度，保持演唱者音区。')
    page=(folder/'index.html').read_text(encoding='utf-8')
    page=page.replace('人声按整曲综合响度匹配到参考人声，再用所有版本共同的静态增益预留峰值余量。未增加混响、压缩或美化。',
                      '三个成品按整曲综合响度匹配，避免音量差影响判断。仅 C 加入轻度 EQ、压缩、短混响及伴奏避让。')
    page=page.replace('</h1>','</h1><p><a href="README.txt">纯文本说明</a></p>',1)
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'README.txt').write_text('青花瓷：节奏保护与轻混音\n\nA：旧节奏、中档修音。\nB：新节奏保护、中档修音，关闭混音效果。\nC：与 B 同一修后人声，开启 EQ、轻压缩、短混响及伴奏避让。\n\n全部合并同一伴奏，时长 3 分 57.2 秒，整曲响度匹配；保留局部动态差异。\n节奏仅接受可靠且局部偏差不超过 ±150ms 的匹配；超限、漏唱或模糊片段保留原位，不跨禁止区域拉伸。\n识别依赖音高和起音特征，并不保证理解所有歌词或漏唱情况。\n\n新节奏本曲接受 56 个起音、跳过 98 个，最大曲线移动约 145ms。\n混音预设较轻，适用于先试听再判断；不能保证所有音色都适合相同 EQ。\n\nGUI 成品不强制使用本试听页的匹配响度，成品混音目标 -16 LUFS，峰值余量优先。\n',encoding='utf-8')
    (folder/'report.json').write_text(json.dumps({'source':str(mixed_path),'measurements':final,'rhythm':report['rhythm']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
