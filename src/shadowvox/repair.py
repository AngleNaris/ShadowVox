"""对明显数字削波的干声做保守修复。"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio


def repair_clipping(samples, sr):
    samples = np.asarray(samples,dtype=np.float32)
    at_limit = np.abs(samples) >= .999
    flat = np.abs(np.diff(samples)) < 1e-5
    runs = int(np.count_nonzero(at_limit[:-2] & at_limit[1:-1] & at_limit[2:] & flat[:-1] & flat[1:]))
    # 很少的顶点也可能只是有损解码过冲；仅明显、持续触顶才自动处理。
    detected = runs >= max(8,int(len(samples)*.00005))
    if not detected:
        return samples,{"applied":False,"clipped_plateau_runs":runs}
    with tempfile.TemporaryDirectory(prefix="shadowvox-declip-") as folder:
        source, target = Path(folder)/"input.wav", Path(folder)/"output.wav"
        sf.write(source,samples,sr,subtype="FLOAT")
        subprocess.run([audio._ffmpeg_exe(),"-v","error","-y","-i",str(source),
                        "-af","adeclip","-c:a","pcm_f32le",str(target)],
                       capture_output=True,check=True,**audio.subprocess_options())
        restored = sf.read(target,dtype="float32")[0]
    if len(restored) < len(samples):
        restored = np.pad(restored,(0,len(samples)-len(restored)))
    restored = restored[:len(samples)]
    # 修复后峰值可能高于满幅；整体降电平以保留恢复的波形并适配后续模型。
    peak = audio.peak(restored)
    gain = min(1.,.98/peak) if peak > 1. else 1.
    restored *= gain
    return restored,{"applied":True,"clipped_plateau_runs":runs,"method":"ffmpeg adeclip",
                     "output_gain_db":float(20*np.log10(gain))}
