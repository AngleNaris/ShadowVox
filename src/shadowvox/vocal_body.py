"""已选 B 的低频增厚：低八度、原基频相干分量及轻第三分量。

各辅轨按整曲活动段标定一次，再与主声汇总；不额外处理主声的音准或共振峰。
这里的“第二/第三”以低八度为基准，不是对整条人声施加饱和失真。
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import butter, hilbert, sosfiltfilt

from . import audio, psola

PRESET = "B_light_third"
OCTAVE_DB = -12.
SECOND_DB = -9.
THIRD_DB = -15.
AUX_MAKEUP_DB = .35
SOURCE_TRIM_DB = -.19


def _gate(sound, pitch, n, sr, centered=False):
    selected = pitch.selected_array
    voiced = (selected['frequency'] > 0) & (selected['strength'] > .6)
    times = np.arange(n)/sr + (sound.x1 if centered else 0.)
    gate = np.interp(times, pitch.xs(), voiced.astype(float), left=0., right=0.)
    # 先转 float32，与已选试听保持相同的门控包络。
    return audio.envelope(gate.astype(np.float32), sr, .030, .080).astype(np.float32)


def _source_cycles(sound, n, sr):
    manipulation = call(sound, 'To Manipulation', .010, 100, 700)
    pulses = call(manipulation, 'Extract pulses')
    count = call(pulses, 'Get number of points')
    times = np.array([call(pulses, 'Get time from index', i) for i in range(1, count+1)])
    runs = np.split(times, np.flatnonzero(np.diff(times) > .012)+1)
    phase = np.zeros(n, dtype=np.float64)
    region = np.zeros(n, dtype=np.float32)
    used = 0
    for points in runs:
        if len(points) < 4:
            continue
        first = max(0, int(np.ceil((points[0]-sound.x1)*sr)))
        last = min(n, int(np.floor((points[-1]-sound.x1)*sr))+1)
        if last <= first:
            continue
        grid = sound.x1+np.arange(first, last)/sr
        # 每个原声周期推进 pi；在无声边界重新起相，避免跨换气插值。
        phase[first:last] = np.interp(grid, points, np.arange(len(points))*np.pi)
        weight = np.ones(last-first, dtype=np.float32)
        fade = min(round(.006*sr), len(weight)//2)
        if fade:
            ramp = np.sin(np.linspace(0, np.pi/2, fade))**2
            weight[:fade], weight[-fade:] = ramp, ramp[::-1]
        region[first:last] = weight
        used += 1
    return phase, region, {'source_pulse_count': int(count), 'used_voiced_runs': used}


def _second_partial(dry, sub, phase, region, gate, sr):
    cosine, sine = np.cos(2*phase), np.sin(2*phase)
    lowpass = butter(2, 25, btype='lowpass', fs=sr, output='sos')
    i_part = sosfiltfilt(lowpass, dry*cosine*region)
    q_part = sosfiltfilt(lowpass, dry*sine*region)
    carrier = (i_part*cosine+q_part*sine)/np.maximum(np.hypot(i_part, q_part), 1e-8)
    amplitude = np.maximum(sosfiltfilt(lowpass, np.abs(hilbert(sub))), 0)
    result = (carrier*amplitude*region*gate).astype(np.float32)
    return sosfiltfilt(butter(4, [200, 450], btype='bandpass', fs=sr, output='sos'),
                      result).astype(np.float32)


def _third_partial(sub, phase, region, sr):
    lowpass = butter(4, 12, btype='lowpass', fs=sr, output='sos')
    i_part = sosfiltfilt(lowpass, sub*np.cos(phase)*region)
    q_part = sosfiltfilt(lowpass, sub*np.sin(phase)*region)
    magnitude = np.hypot(i_part, q_part)
    carrier = ((i_part-1j*q_part)/np.maximum(magnitude, 1e-12))*np.exp(1j*phase)
    result = (2*magnitude*np.real(carrier**3)*region).astype(np.float32)
    return sosfiltfilt(butter(4, [240, 550], btype='bandpass', fs=sr, output='sos'),
                      result).astype(np.float32)


def _normalization_gain(branch, reference, active, target_db=0., scale=1.):
    current = audio.rms((branch*scale)[active])
    target = audio.rms((reference*scale)[active])*10**(target_db/20)
    if current < 1e-9 or target < 1e-9:
        return 0.
    # 不因某首歌的基频落在返回频带外而大幅放大滤波残余。
    return min(target/current, 10**(12/20))


def process(dry, sr, scale=1., normalization=None):
    """返回汇总人声及标定报告；草稿可传入整曲 normalization，避免分段重估电平。"""
    dry = np.asarray(dry, dtype=np.float32)
    stats = {'enabled': False, 'preset': PRESET, 'normalization': {},
             'octave_vs_main_db': OCTAVE_DB, 'second_vs_octave_db': SECOND_DB,
             'third_vs_octave_db': THIRD_DB, 'aux_makeup_db': AUX_MAKEUP_DB,
             'source_trim_db': SOURCE_TRIM_DB,
             'return_bands_hz': [[75, 190], [200, 450], [240, 550]],
             'normalization_reused': normalization is not None,
             'routing': 'sum_before_all_space_and_bus'}
    if len(dry) < round(.08*sr) or audio.rms(dry) < 1e-8:
        return dry.copy(), stats
    sound = parselmouth.Sound(dry.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=.010, pitch_floor=100, pitch_ceiling=700)
    gate = _gate(sound, pitch, len(dry), sr)
    active = gate > .7
    if not np.any(active):
        return dry.copy(), stats
    phase, region, pulse_stats = _source_cycles(sound, len(dry), sr)
    if not pulse_stats['used_voiced_runs']:
        return dry.copy(), stats
    partial_gate = _gate(sound, pitch, len(dry), sr, centered=True)
    del sound, pitch
    shifted = psola.render(dry, sr, np.full(int(np.ceil(len(dry)/sr*100))+1,
                           -1200., dtype=np.float32), frame_rate=100, floor=100, ceiling=700)
    sub = sosfiltfilt(butter(3, [75, 190], btype='bandpass', fs=sr, output='sos'),
                     shifted).astype(np.float32)*gate
    del shifted
    gains = {}
    gains['octave'] = (float(normalization['octave']) if normalization else
                      _normalization_gain(sub, dry, active, OCTAVE_DB))
    sub *= gains['octave']
    second = _second_partial(dry, sub, phase, region, partial_gate, sr)
    third = _third_partial(sub, phase, region, sr)
    active = partial_gate > .7
    for name, branch in (('second', second), ('third', third)):
        gains[name] = (float(normalization[name]) if normalization else
                       _normalization_gain(branch, sub, active, scale=scale))
        branch *= gains[name]
    auxiliary_gain = 10**(AUX_MAKEUP_DB/20)
    source_gain = 10**(SOURCE_TRIM_DB/20)
    auxiliary = (sub*auxiliary_gain+second*10**(SECOND_DB/20)*auxiliary_gain)*source_gain
    summed = (dry*source_gain+auxiliary).astype(np.float32)
    summed += third*(10**(THIRD_DB/20)*auxiliary_gain*source_gain)
    stats.update(enabled=True, normalization=gains, active_seconds=float(np.sum(active)/sr),
                 **pulse_stats)
    return summed, stats
