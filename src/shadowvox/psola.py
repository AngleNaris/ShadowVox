"""TD-PSOLA 渲染（Praat / parselmouth）：直接在原始波形上按音分曲线移调。

按 Praat 分析的音高点做相对移调，未修音区域逐采样旁路。
修音区域仍经过周期拼接；质量依赖周期标注，气声或误检处可能出现瑕疵。
"""
from typing import NamedTuple

import numpy as np
import parselmouth
from parselmouth.praat import call

from .natural import runs


class _Transition(NamedTuple):
    start: float
    end: float
    period_in: float
    period_out: float
    fade_in: float
    fade_out: float
    glide: float


def _smoothstep(fraction):
    x = np.clip(fraction, 0., 1.)
    return x*x*(3.-2.*x)


def _transition(start, end, pitch_in, pitch_out):
    duration = end-start
    period_in, period_out = 1./pitch_in, 1./pitch_out
    # Leave at least a quarter of a short note at the requested shift.
    fade = min(.016, .10*duration)
    return _Transition(start, end, period_in, period_out, min(fade, period_in),
                       min(fade, period_out), min(.045, .24*duration))


def _pitch_changes(times, requested, transitions):
    cents = np.zeros_like(times)
    for t in transitions:
        first, last = np.searchsorted(times, (t.start, t.end))
        window = times[first:last]
        cents[first:last] = requested[first:last] * (
            _smoothstep((window-t.start-t.fade_in)/t.glide) *
            _smoothstep((t.end-window-t.fade_out)/t.glide))
    return cents


def _phase_lag(dry, wet, center, period):
    # Compare about one and a half periods at the seam, searching half a period in either direction.
    radius, width = round(period/2), round(1.5*period)
    start = max(radius, center-width//2)
    stop = min(len(dry)-radius, start+width)
    if stop-start < 12:
        return 0
    reference = dry[start:stop].astype(np.float64)
    reference -= np.mean(reference)
    energy = np.dot(reference, reference)
    if energy <= 1e-10:
        return 0
    search = wet[start-radius:stop+radius]
    length = len(reference)
    total = np.r_[0., np.cumsum(search)]
    power = np.r_[0., np.cumsum(search*search)]
    sums = total[length:] - total[:-length]
    variance = power[length:] - power[:-length] - sums*sums/length
    correlation = np.correlate(search, reference) / np.sqrt(energy*np.maximum(variance, 1e-12))
    best = int(np.argmax(correlation))
    if correlation[radius] >= .40 or correlation[best] < .70:
        return 0
    return best-radius


def _match_seam(dry, wet, indices, segment, fade, period, glide, sr):
    # Keep a constant shift through the short blend, then restore timing before the note's center.
    fade = max(1, round(fade*sr))
    period *= sr
    length = min(len(indices)//2, fade+round(min(.035*sr, 5.*period, glide*sr)))
    if length-fade < 1.5*period:
        return
    lag = _phase_lag(dry, wet, int(indices[fade//2]), period)
    # Limit the local speed change while releasing the lag.
    if lag and 1.5*abs(lag) <= .25*(length-fade):
        d = np.arange(length, dtype=np.float64)
        position = np.clip(indices[:length] + lag*(1.-_smoothstep((d-fade)/(length-fade))), 0, len(wet)-1)
        left = position.astype(np.intp)
        right = np.minimum(left+1, len(wet)-1)
        segment[:length] = wet[left] + (wet[right]-wet[left])*(position-left)


def render(audio, sr, shift_cents, frame_rate=100, floor=65, ceiling=1100):
    """audio：单声道；shift_cents：frame_rate 网格上的移调量。返回同长波形。"""
    raw = np.asarray(audio, dtype=np.float32)
    shifts = np.asarray(shift_cents, dtype=np.float64)
    requested = [(lo/frame_rate, min(hi/frame_rate, len(raw)/sr))
                 for lo, hi in runs(np.abs(shifts) > 1e-3) if hi-lo >= 2]
    if not requested or requested[0][0] >= len(raw)/sr:
        return raw.copy()
    sound = parselmouth.Sound(raw.astype(np.float64), sampling_frequency=sr)
    manipulation = call(sound, "To Manipulation", 0.01, floor, ceiling)
    source = call(manipulation, "Extract pitch tier")
    count = call(source, "Get number of points")
    if count == 0:
        return raw.copy()
    points = np.array([(call(source, "Get time from index", i), call(source, "Get value at index", i))
                       for i in range(1, count+1)])
    source_times, source_hz = points.T
    # Use the same pulse gap as Praat's overlap-add to avoid changing unvoiced material.
    pulses = call(manipulation, "Extract pulses")
    pulse_times = np.array([call(pulses, "Get time from index", i)
                            for i in range(1, call(pulses, "Get number of points")+1)])
    voiced = [(part[0], part[-1]) for part in np.split(pulse_times, np.flatnonzero(np.diff(pulse_times) > .02)+1)
              if len(part) >= 2]
    transitions = []
    cursor = 0
    for start, end in requested:
        while cursor < len(voiced) and voiced[cursor][1] <= start:
            cursor += 1
        for i in range(cursor, len(voiced)):
            a, b = voiced[i]
            if a >= end:
                break
            lo, hi = max(a, start), min(b, end)
            if hi-lo >= 1./frame_rate:
                pitch_in, pitch_out = np.interp([lo, hi], source_times, source_hz)
                transitions.append(_transition(lo, hi, pitch_in, pitch_out))
    if not transitions:
        return raw.copy()

    # Add knots inside both glides so a short note can reach its full correction between analysis frames.
    knots = []
    for t in transitions:
        glide = np.linspace(0., t.glide, 5)
        knots.extend(np.r_[t.start, t.start+t.fade_in+glide, t.end-t.fade_out-glide, t.end])
    times = np.unique(np.r_[source_times, knots])
    hz = np.interp(times, source_times, source_hz)
    grid = np.arange(len(shifts)+1) / frame_rate
    cents = _pitch_changes(times, np.interp(times, grid, np.r_[shifts, 0.], left=0., right=0.), transitions)
    target = call("Create PitchTier", "target", sound.xmin, sound.xmax)
    for time, value in zip(times, hz*2**(cents/1200.)):
        call(target, "Add point", time, value)
    call([target, manipulation], "Replace pitch tier")
    wet = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    out = raw.copy()
    for t in transitions:
        first = max(0, int(np.ceil(t.start*sr)))
        last = min(len(wet), len(raw), int(np.ceil(t.end*sr)))
        indices = np.arange(first, last)
        segment = wet[first:last].copy()
        _match_seam(raw, wet, indices, segment, t.fade_in, t.period_in, t.glide, sr)
        _match_seam(raw, wet, indices[::-1], segment[::-1], t.fade_out, t.period_out, t.glide, sr)
        sample_times = indices/sr
        blend = _smoothstep((sample_times-t.start)/t.fade_in)*_smoothstep((t.end-sample_times)/t.fade_out)
        out[first:last] += (segment-raw[first:last])*blend
    return out
