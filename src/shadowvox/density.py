"""ZL Compressor 四段母带密度压缩，互补分频和段落阈值沿用 ShadowBuster。"""
import threading

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from . import config as C
from .vst_worker import render_isolated

PLUGIN = C.ROOT / 'plugins/ZL Compressor.vst3/Contents/x86_64-win/ZL Compressor.vst3'
CROSSOVERS = (120., 2000., 10000.)
BANDS = ((-22., 2.3, 2.), (-22., 2.6, 3.), (-22., 2.6, 2.5), (-30., 3., 3.))
BLOCK = 2048


def _split(mixed, sr):
    rest = mixed
    for hz in CROSSOVERS:
        low = sosfiltfilt(butter(2,min(hz,sr*.45),fs=sr,output='sos'),rest,axis=-1).astype(np.float32)
        yield low
        rest = rest-low
    yield rest


def _compressor(ratio, attack):
    from pedalboard import load_plugin
    if not PLUGIN.is_file():
        raise RuntimeError('缺少 ZL Compressor 0.6.1，请运行 python tools/fetch_zl.py 或使用完整安装包')
    fx = load_plugin(str(PLUGIN))
    fx.compressor_style_compressor_style = 'Optical'
    fx.ratio_ratio = ratio
    fx.attack_attack = attack
    fx.release_release = 100.
    fx.knee_width_knee_width = 0.
    fx.side_stereo_mode_side_stereo_mode = 'Left Right Max'
    fx.side_stereo_link_side_stereo_link = 100.
    return fx


def densify(mixed, sr, input_lufs, amount, reference=None, *, checkpoint=None):
    """复用整曲的输入响度和分段基准，草稿仍按片段检测局部变化。"""
    amount = float(np.clip(amount,0.,1.))
    stats = {'amount':amount,'applied':False}
    if not amount or mixed.shape[-1] < sr:
        return mixed, stats
    if threading.current_thread() is not threading.main_thread():
        return render_isolated('--density-worker', mixed, sr=sr, input_lufs=input_lufs,
                               amount=amount, reference=reference, checkpoint=checkpoint)
    if reference and reference.get('applied'):
        input_lufs = reference['input_lufs']
        band_levels = reference['band_levels_db']
    else:
        band_levels = None
    threshold_shift = input_lufs + 8.68 + 18. - 6.*amount
    starts = np.arange(0,mixed.shape[-1],BLOCK)
    centers = np.minimum(starts+BLOCK//2,mixed.shape[-1]-1)
    result = np.zeros_like(mixed)
    levels, reductions = [], []
    for index,(band,(threshold,ratio,attack)) in enumerate(zip(_split(mixed,sr),BANDS)):
        power = np.mean(np.square(band,dtype=np.float64),axis=0)
        level_db = 10*np.log10(max(float(power.mean()),1e-20))
        levels.append(level_db if band_levels is None else band_levels[index])
        section = uniform_filter1d(power,round(3.*sr),mode='nearest')[centers]
        follow = np.clip(10*np.log10(np.maximum(section,1e-20))-levels[-1],-12.,6.)
        fx = _compressor(ratio,attack)
        rendered = np.empty_like(band)
        for block,start in enumerate(starts):
            fx.threshold_db_threshold_db = float(np.clip(threshold+threshold_shift+follow[block],-100.,0.))
            chunk = np.ascontiguousarray(band[:,start:start+BLOCK],dtype=np.float32)
            # 每个频段新建实例，保持分块状态并允许 Qt 后台线程处理。
            rendered[:,start:start+BLOCK] = fx.process(chunk,sr,reset=False)
        result += rendered
        hop = round(sr*.05)
        count = band.shape[-1]//hop
        original = power[:count*hop].reshape(count,hop).mean(axis=1)
        output = np.mean(np.square(rendered[:,:count*hop],dtype=np.float64)
                         .reshape(2,count,hop),axis=(0,2))
        active = original > 1e-10
        drop = 10*np.log10(original[active]/np.maximum(output[active],1e-30))
        reductions.append(round(max(0.,float(np.percentile(drop,95))),3) if drop.size else 0.)
    stats.update(applied=True,engine='ZL Compressor 0.6.1',style='Optical',
                 input_lufs=input_lufs,band_levels_db=levels,threshold_shift_db=threshold_shift,
                 crossovers_hz=list(CROSSOVERS),band_reduction_p95_db=reductions)
    return result, stats
