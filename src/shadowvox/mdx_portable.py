"""Kim Vocal 2 的轻量 ONNX 推理，不需要随 EXE 分发 PyTorch。

STFT 约定与 MDX 模型保持一致：periodic Hann、center reflect、未归一 FFT、
立体声实部/虚部四通道。模型参数来自 models/mdx_model_data.json。
"""
import numpy as np
import onnxruntime as ort
from scipy.fft import rfft, irfft
from scipy.signal import get_window

from . import audio, config as C
from .vocoder import _default_providers

FFT = 7680
HOP = 1024
FREQ = 3072
FRAMES = 256
CHUNK = HOP*(FRAMES-1)
WINDOW = get_window("hann",FFT,fftbins=True).astype(np.float32)


def stft(x):
    padded = np.pad(x,((0,0),(FFT//2,FFT//2)),mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded,FFT,axis=-1)[:,::HOP,:]
    z = rfft(frames*WINDOW,axis=-1).transpose(0,2,1)[:,:FREQ]
    return np.stack((z.real,z.imag),axis=1).reshape(1,4,FREQ,-1).astype(np.float32)


def istft(spectrum):
    n_frames = spectrum.shape[-1]
    parts = spectrum.reshape(2,2,FREQ,n_frames)
    z = parts[:,0]+1j*parts[:,1]
    z = np.pad(z,((0,0),(0,FFT//2+1-FREQ),(0,0)))
    frames = irfft(z.transpose(0,2,1),n=FFT,axis=-1)*WINDOW
    n = (n_frames-1)*HOP+FFT
    out = np.zeros((2,n),np.float32);weight=np.zeros(n,np.float32)
    for i in range(n_frames):
        a=i*HOP
        out[:,a:a+FFT]+=frames[:,i]
        weight[a:a+FFT]+=WINDOW**2
    out /= np.maximum(weight,1e-8)
    return out[:,FFT//2:-FFT//2]


def separate(song_path,log=print):
    mix,_ = audio.load(song_path,sr=C.VOC_SR,channels=2)
    peak = audio.peak(mix)
    if peak < 1e-7:
        raise ValueError("原曲没有可用声音")
    normalized = mix/peak
    trim=FFT//2; gen=CHUNK-2*trim
    pad=gen+trim-(mix.shape[-1]%gen)
    padded=np.pad(normalized,((0,0),(trim,pad)))
    out=np.zeros_like(padded);weight=np.zeros(padded.shape[-1],np.float32)
    session=ort.InferenceSession(str(C.MODELS/"Kim_Vocal_2.onnx"),providers=_default_providers())
    log(f"[分离] Kim Vocal 2 · {session.get_providers()[0]}")
    step=int(CHUNK*.75)
    starts=list(range(0,padded.shape[-1],step))
    for k,start in enumerate(starts):
        size=min(CHUNK,padded.shape[-1]-start)
        chunk=np.pad(padded[:,start:start+size],((0,0),(0,CHUNK-size)))
        spec=stft(chunk);spec[:,:,:3,:]=0
        predicted=session.run(None,{"input":spec})[0]
        wave=istft(predicted)[:,:size]
        window=np.hanning(size).astype(np.float32)
        out[:,start:start+size]+=wave*window
        weight[start:start+size]+=window
        log(f"[分离] {k+1}/{len(starts)}")
    out /= np.maximum(weight,1e-8)
    vocals=out[:,trim:trim+mix.shape[-1]]*peak
    inst=mix-vocals*1.009
    return {"vocals":vocals.mean(axis=0),"instrumental":inst}
