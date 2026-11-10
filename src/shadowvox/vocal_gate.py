"""在原唱时间轴上保留唱段，平滑静音演唱干声的其余区间。"""
import numpy as np

from . import config as C
from .natural import runs


def reference_intervals(vocal, f0, sr):
    """返回以原唱采样位置表示的唱段，覆盖字头、尾音及短停顿。"""
    hop = round(sr/C.F0_FRAME_RATE)
    padded = np.pad(vocal,(0,(-len(vocal))%hop))
    rms = np.sqrt(np.mean(np.square(padded.reshape(-1,hop),dtype=np.float64),axis=1))
    voiced = np.zeros(len(rms),dtype=bool)
    size = min(len(voiced),len(f0))
    voiced[:size] = f0[:size] > 0
    voiced &= rms > .0015
    available = rms[voiced] if voiced.any() else rms[rms > .003]
    if not len(available):
        return []
    typical = np.percentile(available,60)
    active = voiced & (rms >= max(.0015,.08*typical))
    # 允许持续的无音高唱段，短促残留由附近唱段的保护边界覆盖。
    for start,end in runs(rms >= max(.003,.32*typical)):
        if end-start >= 6:
            active[start:end] = True

    intervals = []
    for start,end in runs(active):
        lo = max(0,int(start*hop)-round(.09*sr))
        hi = min(len(vocal),int(end*hop)+round(.13*sr))
        if intervals and lo <= intervals[-1][1]:
            lo, previous_end = intervals.pop()
            hi = max(hi,previous_end)
        intervals.append((lo,hi))
    return intervals


def mute_gaps(vocal, intervals, sr, start=0):
    """start 是本段首样本在原唱时间轴的位置；整轨和局部片段使用相同淡入淡出。"""
    result = np.zeros_like(vocal)
    fade = max(1,round(.02*sr))
    kept = 0
    for begin,end in intervals:
        lo,hi = max(0,begin-start),min(len(vocal),end-start)
        if hi <= lo:
            continue
        samples = np.arange(start+lo,start+hi)
        edge = np.minimum(samples-begin,end-1-samples)
        blend = np.clip(edge/fade,0.,1.)
        gain = .5-.5*np.cos(np.pi*blend)
        result[lo:hi] = vocal[lo:hi]*gain
        kept += hi-lo
    return result, kept
