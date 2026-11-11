"""音区保护：乐句/持续音区决定八度，去颤音的音高中心差决定修正量。"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, median_filter, uniform_filter1d

from .natural import runs

LARGE_NOTE_LIMIT_CENTS = 300.


def fold_register(dry, reference, voiced):
    """100fps 音分。250ms 以上空隙分句；音区改变需持续至少 300ms。

    同一片段共享八度关系。450 音分用于判定音区证据，音符中心保护在折叠后应用。
    """
    octave = np.zeros(len(dry), dtype=np.int32)
    indices = np.flatnonzero(voiced)
    if not len(indices):
        return reference.copy(), np.zeros(len(dry), bool), octave
    breaks = np.r_[0, np.flatnonzero(np.diff(indices) > 25) + 1, len(indices)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        active = indices[a:b]
        start, end = active[0], active[-1] + 1
        difference = np.interp(np.arange(start, end), active, dry[active] - reference[active])
        smooth = median_filter(difference, size=21, mode="nearest")
        candidates = np.rint(smooth / 1200).astype(np.int32)
        phrase_octave = int(np.rint(np.median(difference) / 1200))
        local = np.full(end-start, phrase_octave, dtype=np.int32)
        bounds = np.r_[0, np.flatnonzero(np.diff(candidates)) + 1, len(candidates)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            k = candidates[lo]
            good = voiced[start+lo:start+hi] & (np.abs(difference[lo:hi] - 1200*k) <= 450)
            if hi-lo >= 30 and good.sum() >= .7*(hi-lo):
                local[lo:hi] = k
        octave[start:end] = local
    target = reference + 1200*octave
    reliable = voiced & (np.abs(target-dry) <= 450)
    return target, reliable, octave


def register_shift(dry, reference, voiced, strength, max_shift_cents=450, detail=0.):
    target, reliable, octave = fold_register(dry, reference, voiced)
    difference = target - dry
    shift = np.zeros(len(dry), dtype=np.float64)
    detail = float(np.clip(detail,0.,1.))
    for start, end in runs(reliable):
        if end-start < round(10-4*detail):
            continue
        shift[start:end], _, _ = _slow_center(difference[start:end], detail, max_shift_cents, strength)
    shift[~reliable] = 0
    large_notes = [(lo, hi) for _, _, lo, hi in _note_ranges(dry, reference, voiced, difference, octave)
                   if hi-lo >= 8 and round(abs(np.median(target[lo:hi])-np.median(dry[lo:hi])), 1)
                   > LARGE_NOTE_LIMIT_CENTS]
    protected, _ = _protect_large_notes(shift, large_notes)
    reliable[protected] = False
    return shift, reliable, octave


def _slow_center(difference, detail, max_shift_cents, strength=1.):
    # 110ms→150ms 中值级联压掉两轨颤音的差异，再做 50ms 过渡。
    first, second = 11-2*round(2*detail), 15-2*round(3*detail)
    delta = median_filter(median_filter(difference, size=first, mode="nearest"),
                          size=second, mode="nearest")
    delta = uniform_filter1d(delta, size=5, mode="nearest")
    deviation = median_filter(np.abs(difference-delta), size=21, mode="nearest")
    confidence = np.clip((160-deviation)/100, 0, 1)
    shift = gaussian_filter1d(np.clip(delta, -max_shift_cents, max_shift_cents)*strength*confidence,
                              sigma=1, mode="constant")
    return shift, delta, confidence


def _pitch_boundaries(track):
    """Use adjacent 50 ms medians to find persistent steps without quantizing notes."""
    window = 5
    if len(track) < 2 * window:
        return np.empty(0, dtype=int)
    stable = median_filter(track, size=3, mode="nearest")
    centers = np.median(sliding_window_view(stable, window), axis=1)
    change = centers[window:] - centers[:-window]
    # Opposite steps in quick succession (a short note) remain separate.
    boundaries = []
    for mask in (change >= 85., change <= -85.):
        for start, end in runs(mask):
            indices = np.arange(start, end) + window
            contrast = np.abs(change[start:end])
            # Break ties at the actual step instead of the edge of the window.
            jump = np.abs(stable[indices] - stable[indices-1])
            boundaries.append(indices[np.argmax(contrast + .25 * jump)])
    return np.asarray(boundaries, dtype=int)


def _note_ranges(dry, reference, voiced, difference, octave):
    for start, end in runs(voiced):
        steps = np.r_[_pitch_boundaries(dry[start:end]), _pitch_boundaries(reference[start:end])]
        residual_steps = [i for i in _pitch_boundaries(difference[start:end]) if np.any(np.abs(steps-i) <= 2)]
        octave_bounds = np.flatnonzero(np.diff(octave[start:end])) + 1
        bounds = np.unique([0, *residual_steps, *octave_bounds, end-start]) + start
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            yield start, end, lo, hi


def _protect_large_notes(shift, notes):
    """Clear the whole note and ease the transition across up to two neighboring frames."""
    protected = np.zeros(len(shift), dtype=bool)
    count = 0
    for lo, hi in notes:
        if np.any(shift[lo:hi]):
            protected[lo:hi] = True
            count += 1
    if count:
        shift *= np.minimum(distance_transform_edt(~protected) / 3., 1.)
    return protected, count


def _short_note_needs_plan(difference, legacy, start, end, lo, hi, delta):
    # Keep an already effective slow-center plan. Replace a missed center or leakage
    # into a neighboring region that was already close to its own target.
    if not np.any(legacy[lo:hi]) or abs(delta - np.median(legacy[lo:hi])) >= 35.:
        return True
    if abs(delta) < 100.:
        return False
    for a, b, edge in ((max(start, lo-6), lo, slice(max(start, lo-2), lo)),
                       (hi, min(end, hi+6), slice(hi, min(end, hi+2)))):
        if b-a < 4:
            continue
        neighbor = np.median(difference[a:b])
        if abs(neighbor) < 70. and np.max(np.abs(legacy[edge] - neighbor)) >= 50.:
            return True
    return False


def correspondence_shift(dry, reference, voiced, strength, max_shift_cents=450, detail=0.):
    """Plan stable short notes and slow centers in the singer's octave."""
    target, register_support, octave = fold_register(dry, reference, voiced)
    difference = target - dry
    shift = np.zeros(len(dry), dtype=np.float64)
    planned = np.zeros(len(dry), dtype=bool)
    short = np.zeros(len(dry), dtype=bool)
    conflicts = voiced & (np.abs(difference) > 850.)
    remaining = register_support.copy()
    detail = float(np.clip(detail, 0., 1.))
    minimum = round(8 - 2 * detail)
    slow_minimum = round(10 - 4 * detail)
    limited = np.zeros(len(dry), dtype=bool)
    in_tune = np.zeros(len(dry), dtype=bool)
    short_planned = np.zeros(len(dry), dtype=bool)
    large_notes = []
    legacy = np.zeros(len(dry), dtype=np.float64)
    for lo, hi in runs(register_support):
        if hi-lo >= slow_minimum:
            legacy[lo:hi], _, _ = _slow_center(difference[lo:hi], detail, max_shift_cents)

    for start, end, lo, hi in _note_ranges(dry, reference, voiced, difference, octave):
        count = int(hi - lo)
        if count < minimum:
            short[lo:hi] = True
            continue
        pair = np.stack((dry[lo:hi], target[lo:hi]))
        centers = np.median(pair, axis=1)
        delta = float(centers[1] - centers[0])
        if round(abs(delta), 1) > LARGE_NOTE_LIMIT_CENTS:
            large_notes.append((lo, hi))
        if count > 15 and abs(delta) <= 450.:
            continue
        deviations = np.abs(pair - centers[:, None])
        if np.quantile(deviations, .8, axis=1).max() > 90.:
            continue
        local = slice(max(start, lo-30), min(end, hi+30))
        anchored = np.any(register_support[local] & (octave[local] == octave[lo]))
        # A near-octave jump or an unanchored half-octave ambiguity may be a tracker error.
        if abs(delta) > 850. or (abs(delta) > 560. and not anchored):
            conflicts[lo:hi] = True
            remaining[lo:hi] = False
            continue
        if abs(delta) <= 450. and not _short_note_needs_plan(difference, legacy, start, end, lo, hi, delta):
            continue
        supported = np.max(deviations, axis=0) <= 350.
        planned[lo:hi] = supported
        remaining[lo:hi] = False
        if count <= 15:
            short_planned[lo:hi] = supported
        if abs(delta) < 3.:
            in_tune[lo:hi] = supported
            continue
        if abs(delta) > max_shift_cents:
            limited[lo:hi] = supported
        amount = float(np.clip(delta, -max_shift_cents, max_shift_cents))
        for a, b in runs(supported):
            shift[lo+a:lo+b] = gaussian_filter1d(np.full(b-a, amount), sigma=.8, mode="constant")

    for lo, hi in runs(remaining):
        if hi-lo < slow_minimum:
            short[lo:hi] = True
            continue
        planned[lo:hi] = True
        shift[lo:hi], delta, confidence = _slow_center(difference[lo:hi], detail, max_shift_cents)
        in_tune[lo:hi] = np.abs(delta) < 3.
        limited[lo:hi] = (np.abs(delta) > max_shift_cents) & (confidence > 0.)

    protected, protected_notes = _protect_large_notes(shift, large_notes)
    planned[protected] = False
    np.clip(shift, -max_shift_cents, max_shift_cents, out=shift)
    shift *= strength
    large = voiced & (np.abs(difference) > 450.)
    unplanned = voiced & ~planned & ~protected
    coverage = dict(short_frames=int((unplanned & short & ~conflicts).sum()),
                    short_planned_frames=int((planned & short_planned).sum()),
                    unstable_frames=int((unplanned & ~short & ~conflicts).sum()),
                    register_conflict_frames=int((unplanned & conflicts).sum()),
                    limited_frames=int((planned & limited).sum()), in_tune_frames=int((planned & in_tune).sum()),
                    planned_frames=int(planned.sum()), shifted_frames=int(np.count_nonzero(np.abs(shift) > 1e-3)),
                    large_note_protected_notes=protected_notes, large_note_protected_frames=int(protected.sum()),
                    large_error_frames=int(large.sum()), large_error_planned_frames=int((large & planned).sum()))
    return shift, planned, octave, coverage
