"""独立目录 + 精简 PATH 下验收单 EXE，首次分离完整歌曲和原生窗口。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shadowvox.mixing import measure_loudness


def run():
    folder=ROOT/'out/portable_final'/str(int(time.time()))
    folder.mkdir(parents=True)
    source_exe=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'dist/ShadowVox.exe'
    exe=folder/'ShadowVox.exe';shutil.copy2(source_exe,exe)
    previous=json.loads((ROOT/'out/qinghua.report.json').read_text(encoding='utf-8'))
    output=folder/'青花瓷_成品.wav'
    request={'song':previous['song'],'dry':previous['dry'],'output':str(output),
             'work':str(folder/'cache'),'strength':.75,'mix_effects':True,'reuse_stems':False}
    request_path=folder/'request.json'
    request_path.write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
    previews=ROOT/'out/desktop_preview/pink_v3'
    (folder/'preview-request.json').write_text(json.dumps(dict(song=previous['song'],dry=previous['dry'],
        before=str(previews/'before.wav'),after=str(previews/'after.wav')),ensure_ascii=False),encoding='utf-8')
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('QT_QPA_PLATFORM',None)
    env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    env['LOCALAPPDATA']=str(folder/'userdata')
    print('Testing frozen native GUI',flush=True)
    gui=subprocess.run([str(exe),'--gui-check',str(folder)],cwd=folder,env=env,
                       creationflags=subprocess.CREATE_NO_WINDOW,timeout=45)
    assert gui.returncode==0
    state=json.loads((folder/'frozen-gui.json').read_text(encoding='utf-8'))
    assert state['visible'] and state['strength']==75 and state['mix_enabled']
    assert state['playback_enabled'] and state['frames_written']>0 and state['selected']==0
    assert state['position']>64*44100 and state['stream_active'],state
    start=time.monotonic()
    print('Testing frozen full-song pipeline',flush=True)
    result=subprocess.run([str(exe),'--process',str(request_path)],cwd=folder,env=env,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    elapsed=time.monotonic()-start
    assert result.returncode==0,output.with_suffix('.log').read_text(encoding='utf-8')
    report=json.loads(output.with_suffix('.report.json').read_text(encoding='utf-8'))
    assert report['options']['denoise'] and report['denoise']['model']=='DeepFilterNet3'
    assert report['denoise']['attenuation_limit_db']==12
    assert report['mixing']['enabled'] and 'max_deess_db' in report['mixing']
    samples,sr=sf.read(output,dtype='float32',always_2d=True)
    assert samples.shape[1]==2 and np.isfinite(samples).all()
    expected=max(report['song_seconds']+report['rhythm'].get('reference_shift_seconds',0),
                 report['dry_seconds_after_rhythm'])
    assert abs(len(samples)/sr-expected)<.02
    # 冻结程序必须实际走 PSOLA + 整句对齐（numba、内置 Rubber Band 均可用）。
    assert report['options']['renderer']=='psola' and report['rhythm']['algorithm']=='hubertfa-zh+rubberband-r3-v1'
    assert 'lyric_error' not in report['rhythm'],report['rhythm'].get('lyric_error')
    lufs,peak=measure_loudness(output)
    assert peak<=-1.45,(lufs,peak)
    print(f'Audio passed: {elapsed:.1f}s including startup, {lufs} LUFS, {peak} dBTP',flush=True)
    summary={'passed':True,'source_exe':str(source_exe),'exe':str(exe),'exe_bytes':exe.stat().st_size,'output':str(output),
             'wall_seconds_including_startup':round(elapsed,1),'processing_seconds':report['elapsed_seconds'],
             'seconds':len(samples)/sr,'sample_rate':sr,'lufs':lufs,'true_peak_dbtp':peak,
             'rhythm':{k:report['rhythm'][k] for k in ('algorithm','lyric_anchors','rate_min','rate_max')},
             'renderer':report['options']['renderer'],
             'gui':state,'screenshot':str(folder/'frozen-gui.png')}
    (folder/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'out/portable_final/latest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
