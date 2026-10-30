"""傻瓜式修音主流程：原曲 → 分离 → 双轨 F0 → 对齐映射 → 重合成 → 混音落盘。"""
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import __version__, audio, config as C
from .align import estimate_shift, apply_shift
from . import rhythm
from .f0 import FcpeExtractor
from .separate import separate_song, DEFAULT_MODEL
from .vocoder import Vocoder


@dataclass
class Options:
    strength: float = 0.85          # 修正力度 0..1
    max_shift_cents: float = 700.0  # 单帧最大移调
    smooth_ms: float = 0.0          # frame 模式平滑窗；0=逐帧跟随，不代表自然度最佳
    pitch_mode: str = "frame"       # natural 为保留音符内形状的实验模式
    vocal_gain_db: float = 0.0      # 修后干声相对原唱响度的额外增益
    octave: str = "auto"            # auto=折叠到干声自身音区 | match=跟原唱同音区（可整体升降八度）
    rhythm: bool = True             # 入点级节奏对齐（跟原唱对齐，只修入点，不逐字硬贴）
    rhythm_max_rate: float = 0.10   # 局部时间伸缩上限，10%=最多拉伸/压缩 10%
    rhythm_mode: str = "guarded"    # guarded 只修可信入点 | full 整句逐帧贴齐原唱（需 Rubber Band）
    separate_model: str = DEFAULT_MODEL
    reuse_stems: bool = True
    mix: bool = True
    mix_effects: bool = False      # GUI 开关：EQ/轻压缩/混响/避让；伴奏合并独立于此
    comparison: bool = False       # 桌面试听：额外保存未经局部节奏/音高校正的同伴奏版本
    renderer: str = "vocoder"      # vocoder 神经重合成 | psola 原波形 TD-PSOLA | rubberband R3
    denoise: bool = True           # 修音前使用已试听确认的 AI 温和降噪
    mix_space: float = 1.0         # 混响与延时发送的共同倍率
    denoise_strength: float = 1.0  # 原始干声与温和降噪结果之间的比例
    declip: bool = True
    rhythm_strength: float = 1.0
    mix_lowcut_hz: float = 75.0
    mix_eq: float = 1.0
    mix_compression: float = 1.0
    mix_deess: float = 1.0
    mix_reverb: float = 1.0
    mix_delay: float = 1.0
    mix_duck: float = 1.0
    target_lufs: float = -16.0
    true_peak_ceiling: float = -1.5


def _detect_octave(dry_f0, ref_f0, move_seconds):
    """估计干声相对原唱的整体八度差（负数=干声更低）。

    move_seconds 是「干声需移动的量」（正=延后），与 estimate_shift 同语义：
    dry[j] 对应 ref[j + move]，因此 ref 侧下标为 j + lag。
    """
    lag = int(round(move_seconds * C.F0_FRAME_RATE))
    idx = np.arange(len(dry_f0)) + lag
    ok = (dry_f0 > 0) & (idx >= 0) & (idx < len(ref_f0))
    if ok.sum() < 50:
        return 0.0
    r = ref_f0[np.clip(idx, 0, len(ref_f0) - 1)][ok]
    d = dry_f0[ok]
    ok2 = r > 0
    if ok2.sum() < 50:
        return 0.0
    return float(np.median(np.log2(r[ok2] / d[ok2])))


def _key(*paths):
    h = hashlib.sha1()
    for p in paths:
        st = Path(p).stat()
        h.update(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def _load_or_separate(song_path, work_dir, model, reuse, log):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = _key(song_path)[:8]
    vpath = work_dir / f"vocals_{tag}.wav"
    ipath = work_dir / f"inst_{tag}.wav"

    if reuse and vpath.exists() and ipath.exists():
        vocal, inst = audio.load_mono(vpath, C.VOC_SR), audio.load(ipath, sr=C.VOC_SR, channels=2)[0]
        # 缓存校验：人声轨必须真有能量且与原曲同量级；上次跑挂留下的残次品要重算
        if _stems_ok(vocal, inst, song_path):
            log("[分离] 复用缓存")
            return vocal, inst
        log("[分离] 缓存不可用，重新分离")

    t0 = time.time()
    stems = separate_song(song_path, work_dir, model=model, log=log)
    vocal = stems["vocals"]
    inst = stems["instrumental"]
    if inst is None:
        song = audio.load(song_path, sr=C.VOC_SR, channels=2)[0]
        n = min(song.shape[1], len(vocal))
        inst = song[:, :n] - np.vstack([vocal[:n], vocal[:n]])
    audio.save(vpath, vocal, C.VOC_SR)
    audio.save(ipath, inst, C.VOC_SR)
    log(f"[分离] 完成 {time.time() - t0:.1f}s")
    return vocal, inst


def _stems_ok(vocal, inst, song_path):
    """缓存完整性校验：长度接近原曲、人声/伴奏都有能量，且两者不高度相关。"""
    try:
        song = audio.load(song_path, sr=C.VOC_SR, channels=2)[0]
    except Exception:
        return True                     # 原曲读不了就别拦着，交给后面的步骤报错
    want = song.shape[1]
    if abs(len(vocal) - want) > C.VOC_SR * 2 or inst.shape[1] < want * 0.9:
        return False
    v, i = audio.rms(vocal), audio.rms(inst.mean(axis=0))
    if v < 1e-4 or i < 1e-4:
        return False
    # 分离失败时人声轨常等于原曲（相关系数接近 1），正常分离应在 0.5 以下
    n = min(len(vocal), song.shape[1])
    ref = song.mean(axis=0)[:n]
    seg = vocal[:n]
    if np.std(seg) < 1e-9 or np.std(ref) < 1e-9:
        return False
    return abs(float(np.corrcoef(seg, ref)[0, 1])) < 0.9


def run(dry_path, song_path, out_path=None, work_dir=None, opts=None, log=print, capture=None):
    opts = opts or Options()
    dry_path, song_path = Path(dry_path), Path(song_path)
    out_path = Path(out_path) if out_path else C.OUT_DIR / f"{dry_path.stem}_autotuned.wav"
    work_dir = Path(work_dir) if work_dir else C.WORK_DIR / song_path.stem
    report = {"app_version": __version__, "dry": str(dry_path), "song": str(song_path), "options": asdict(opts)}
    if not 0 <= opts.strength <= 1:
        raise ValueError("修音强度必须在 0 到 1 之间")
    if not 0 <= opts.mix_space <= 2:
        raise ValueError("空间感必须在 0% 到 200% 之间")
    if not 0 <= opts.denoise_strength <= 1 or not 0 <= opts.rhythm_strength <= 1:
        raise ValueError("降噪及节奏强度必须在 0% 到 100% 之间")
    targets=(out_path,out_path.with_name(out_path.stem+"_vocal.wav"),out_path.with_suffix('.report.json'))
    if opts.comparison:
        targets += (out_path.with_name(out_path.stem+'_before.wav'),)
    if any(p.resolve() in (dry_path.resolve(),song_path.resolve()) for p in targets):
        raise ValueError("输出不能覆盖原曲或演唱干音")
    if out_path.suffix.lower() != ".wav":
        raise ValueError("成品输出请使用 .wav 扩展名")
    t_all = time.time()

    log("[1/6] 读取干声")
    dry = audio.load_mono(dry_path, C.VOC_SR)
    original_dry = dry.copy() if opts.comparison else None
    if capture is not None:
        capture['dry_input'] = dry.copy()
    report["dry_seconds"] = round(len(dry) / C.VOC_SR, 2)
    from .repair import repair_clipping
    if opts.declip:
        dry, report["declip"] = repair_clipping(dry,C.VOC_SR)
    else:
        report["declip"] = {"applied": False, "disabled": True}
    if report["declip"]["applied"]:
        log("[修复] 修复明显的数字削波")
    if capture is not None:
        capture['dry_before_denoise'] = dry.copy()
    if opts.denoise and opts.denoise_strength:
        from .denoise import process as denoise
        log("[降噪] AI 温和降噪")
        cleaned = denoise(dry, C.VOC_SR)
        if capture is not None:
            capture['denoised_dry'] = cleaned.copy()
        dry = cleaned if opts.denoise_strength == 1 else dry + (cleaned - dry) * opts.denoise_strength
        report["denoise"] = {"enabled": True, "model": "DeepFilterNet3", "attenuation_limit_db": 12,
                             "strength": opts.denoise_strength}
    else:
        report["denoise"] = {"enabled": False}

    log("[2/6] 分离原曲")
    ref_vocal, inst = _load_or_separate(song_path, work_dir, opts.separate_model,
                                        opts.reuse_stems, log)
    if capture is not None:
        capture['reference_original'] = ref_vocal.copy()
        capture['inst_original'] = inst.copy() if inst is not None else None
    n = min(len(dry), len(ref_vocal))
    report["song_seconds"] = round(len(ref_vocal) / C.VOC_SR, 2)
    report["dry_vs_song_ratio"] = round(len(dry) / max(1, len(ref_vocal)), 3)

    log("[3/6] 提取双轨音高")
    fcpe = FcpeExtractor()
    dry_f0 = fcpe.extract(dry, C.VOC_SR)
    ref_f0 = fcpe.extract(ref_vocal, C.VOC_SR)
    if capture is not None:
        capture['input_f0'] = dry_f0.copy()
    report["dry_voiced_ratio"] = round(float((dry_f0 > 0).mean()), 3)
    report["ref_voiced_ratio"] = round(float((ref_f0 > 0).mean()), 3)

    # 节奏对齐：以原唱入点为锚，只修入点，句内时值靠平滑过渡保住。
    # 放在音高之前：时间轴一改，音高曲线必须重提，否则音高与波形错位。
    # 未开启节奏处理时保持原位置，不能靠音高相似度暗中平移或裁掉开头。
    onset_offset = 0.0
    before_move = None
    if opts.rhythm and opts.rhythm_strength and opts.rhythm_mode == "full":
        from .full_align import align
        log('[节奏] 整句对齐到原唱')
        dry, rstat = align(dry, ref_vocal, C.VOC_SR, log=log,
                           strength=opts.rhythm_strength, capture=capture,
                           max_rate=opts.rhythm_max_rate)
        report["rhythm"] = rstat
        reference_shift = rstat.get("reference_shift_seconds", 0.)
        if reference_shift:
            front = round(reference_shift * C.VOC_SR)
            ref_vocal = np.pad(ref_vocal, (front, 0))
            inst = np.pad(inst, ((0, 0), (front, 0)))
            ref_f0 = np.r_[np.zeros(round(reference_shift * C.F0_FRAME_RATE), dtype=ref_f0.dtype), ref_f0]
        onset_offset = 0.0              # 已在原唱时间线上
        before_move = rstat["median_offset_seconds"] + reference_shift
        dry_f0 = fcpe.extract(dry, C.VOC_SR)
    elif opts.rhythm and opts.rhythm_strength:
        log('[节奏] 检查可靠起音并匹配节奏')
        dry, rstat = rhythm.align_to_reference(
            dry, dry_f0, ref_f0, C.VOC_SR, max_rate=opts.rhythm_max_rate, log=log)
        report["rhythm"] = rstat
        # 固定经核验的整体录音偏移；局部变时后重估会挪动应保护的片段。
        onset_offset = rstat.get("placement_offset_seconds")
        if rstat.get("applied"):
            dry_f0 = fcpe.extract(dry, C.VOC_SR)
            # 时间轴已变，偏移必须在新波形上重测，否则摆放会沿用旧值
            if onset_offset is None:
                onset_offset = rhythm.onset_offset(dry_f0, ref_f0)
            report["rhythm"]["offset_after_seconds"] = round(onset_offset, 4)
    report["dry_seconds_after_rhythm"] = round(len(dry) / C.VOC_SR, 2)
    if capture is not None:
        capture['aligned_dry'] = dry.copy()
        capture['dry_f0'] = dry_f0.copy()
        capture['ref_f0'] = ref_f0.copy()

    log("[4/6] 对齐与音高映射")
    # 摆放偏移与音高映射必须用同一个 offset，否则音高修正会参考错位的帧
    # （实测青花瓷：两者错开 40ms，音准从 24 恶化到 40 音分）
    res = estimate_shift(dry_f0, ref_f0, strength=opts.strength,
                         max_shift_cents=opts.max_shift_cents,
                         smooth_ms=opts.smooth_ms,
                         pitch_mode=opts.pitch_mode,
                         offset_override=onset_offset)

    # 八度策略：默认 auto 只在干声本音区内修正（保留演唱者音色）；
    # match 会把干声整体搬到原唱音区，适合“低八度/高八度跟唱”。
    src_f0 = dry_f0
    octave_delta = 0.0
    if opts.octave == "match":
        od = _detect_octave(dry_f0, ref_f0, res.offset_seconds)
        octave_delta = float(np.rint(od))
        if octave_delta != 0.0:
            log(f"[4/6] 干声与原唱相差 {octave_delta:+.0f} 个八度，整体搬运后重新对齐")
            src_f0 = np.where(dry_f0 > 0, dry_f0 * (2.0 ** octave_delta), 0.0).astype(np.float32)
            res = estimate_shift(src_f0, ref_f0, strength=opts.strength,
                                 max_shift_cents=opts.max_shift_cents,
                                 smooth_ms=opts.smooth_ms,
                                 pitch_mode=opts.pitch_mode,
                                 offset_override=onset_offset)
    report["octave_shift"] = octave_delta

    # 不可靠节奏区域也不按错误参考音符强行修音。
    for start,end in report.get("rhythm",{}).get("protected_intervals_seconds",[]):
        res.shift_cents[round(start*C.F0_FRAME_RATE):round(end*C.F0_FRAME_RATE)] = 0
        res.confidence[round(start*C.F0_FRAME_RATE):round(end*C.F0_FRAME_RATE)] = 0
    # 原曲结束后的保留人声没有可用参考，不能沿用最后一个音高。
    res.shift_cents[round(len(ref_vocal) / C.VOC_SR * C.F0_FRAME_RATE):] = 0
    selected = src_f0 > 0
    res.stats["mean_abs_shift_cents"] = float(np.mean(np.abs(res.shift_cents[selected]))) if selected.any() else 0.
    res.stats["max_abs_shift_cents"] = float(np.max(np.abs(res.shift_cents),initial=0))
    res.stats["mean_confidence"] = float(res.confidence[selected].mean()) if selected.any() else 0.

    corr_f0 = apply_shift(src_f0, res.shift_cents)
    report["align"] = {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in res.stats.items()}
    if capture is not None:
        capture['shift_cents'] = res.shift_cents.copy()

    log("[5/6] 修正人声")
    if opts.renderer == "vocoder":
        vocal = Vocoder().synth(dry, corr_f0, progress=lambda p: log(f"[合成] {p:.0%}"))
    else:
        # 波形类渲染器只接受相对移调；含 match 模式的整体八度搬移。
        voiced = (dry_f0 > 0) & (corr_f0 > 0)
        total = np.zeros(len(dry_f0))
        total[voiced] = 1200 * np.log2(corr_f0[voiced] / dry_f0[voiced])
        if opts.renderer == "psola":
            from .psola import render
            vocal = render(dry, C.VOC_SR, total)
        elif opts.renderer == "rubberband":
            from .rubberband import render
            vocal = render(dry, total, C.VOC_SR)[0]
        else:
            raise ValueError(f"未知渲染器: {opts.renderer}")
    report["synth_seconds"] = round(len(vocal) / C.VOC_SR, 2)

    log("[6/6] 混音落盘")
    spf = int(round(C.VOC_SR / C.F0_FRAME_RATE))
    nf = min(len(ref_f0), len(ref_vocal) // spf)
    vmask = np.repeat(ref_f0[:nf] > 0, spf)
    ref_voiced = ref_vocal[: len(vmask)][vmask]
    ref_voiced_rms = audio.rms(ref_voiced) or audio.rms(ref_vocal)
    unscaled_rms = audio.rms(vocal)
    vocal = audio.match_rms(vocal, ref_voiced_rms)
    if capture is not None:
        capture['vocal_normalization'] = (audio.rms(vocal) / unscaled_rms if unscaled_rms else 1.)
    vocal = vocal * (10 ** (opts.vocal_gain_db / 20.0))
    report["vocal_rms_dbfs"] = round(20 * np.log10(max(audio.rms(vocal), 1e-9)), 1)

    # res.offset_seconds = 干声比原唱晚多少（正=晚）；_place 要的是「内容平移量」
    # （正=延后），即把干声往回挪，故取负号。
    # 已用两套已知答案交叉验证：
    #   selftest 干声晚 0.373s → offset +0.373 → move -0.373 → 对齐（残留 0.02s）
    #   青花瓷 干声早   0.086s → offset -0.086 → move +0.086 → 音高误差 47.4 音分（≈最优 47.3）
    #
    # 若做了节奏对齐，用**入点偏移**而不是音高偏移：两者相差约 25ms，
    # 用户优先级是「入点 > 中间 > 出点」，音高是逐帧局部修正，这点差只值 ~3 音分。
    if onset_offset is not None:
        move = -onset_offset
    else:
        move = -res.offset_seconds
    song_len = inst.shape[1] if inst is not None else len(ref_vocal)
    # 原唱缺失内容时，保留的演唱可能超出原曲末尾；伴奏补静音而不是截字。
    if opts.rhythm and opts.rhythm_strength and opts.rhythm_mode == "full" and len(vocal) > song_len:
        song_len = len(vocal)
        if inst is not None:
            inst = np.pad(inst, ((0, 0), (0, song_len - inst.shape[1])))
    total = max(song_len, len(vocal) + int(abs(move) * C.VOC_SR))
    if inst is not None:
        total = max(total, inst.shape[1])
    placed = _place(vocal, move, total, C.VOC_SR)
    if len(placed) > song_len:
        trimmed = (len(placed) - song_len) / C.VOC_SR
        placed = placed[:song_len]
        report["vocal_trimmed_seconds"] = round(trimmed, 2)
    if capture is not None:
        capture.update(inst=inst.copy() if inst is not None else None, vocal=placed.copy(),
                       reference=ref_vocal.copy(), move=move, ref_voiced_rms=ref_voiced_rms,
                       options=asdict(opts))
        log('[报告] 分析实际修正后人声')
        capture['rendered_f0'] = fcpe.extract(placed, C.VOC_SR)

    vocal_path = out_path.with_name(out_path.stem + "_vocal.wav")
    audio.save(vocal_path, placed, C.VOC_SR)

    if opts.comparison and inst is not None:
        original = audio.match_rms(original_dry, ref_voiced_rms)
        original *= 10 ** (opts.vocal_gain_db / 20.0)
        original = _place(original, move if before_move is None else before_move, song_len, C.VOC_SR)
        before = inst[:, :song_len] + original[None, :]
        before *= min(1., 10**(-1.5/20) / max(audio.peak(before), 1e-9))
        before_path = out_path.with_name(out_path.stem + '_before.wav')
        audio.save(before_path, before, C.VOC_SR)
        report['comparison_before'] = str(before_path)
        report['comparison_global_move_seconds'] = float(move if before_move is None else before_move)

    if opts.mix and inst is not None:
        stereo = np.vstack([placed, placed])
        n_mix = min(stereo.shape[1], inst.shape[1])
        if opts.mix_effects:
            from .mixing import enhance, master
            log("[混音] 人声质感、原唱比例参考、空间效果与总线粘合")
            mixed, effect_stats = enhance(placed[:n_mix],inst[:,:n_mix],C.VOC_SR,space=opts.mix_space,
                lowcut_hz=opts.mix_lowcut_hz,eq=opts.mix_eq,compression=opts.mix_compression,
                deessing=opts.mix_deess,reverb=opts.mix_reverb,delay_amount=opts.mix_delay,duck=opts.mix_duck,
                reference_vocal=ref_vocal[:n_mix],vocal_gain_db=opts.vocal_gain_db,capture=capture)
            mixed, master_stats = master(mixed,C.VOC_SR,target_lufs=opts.target_lufs,
                                         true_peak_ceiling=opts.true_peak_ceiling,apply_glue=True)
            report["mixing"] = {**effect_stats,**master_stats}
        else:
            mixed = inst[:, :n_mix] + stereo[:, :n_mix]
            report["mixing"] = {"enabled":False}
        # 峰值保护：留 1dB 余量，避免重采样/有损编码后削顶
        pk = audio.peak(mixed)
        ceiling = 10 ** (-1.0 / 20.0)        # ≈0.891
        if pk > ceiling:
            mixed = mixed * (ceiling / pk)
            report["peak_limited_db"] = round(float(20 * np.log10(ceiling / pk)), 2)
        audio.save(out_path, mixed, C.VOC_SR)
        report["output"] = str(out_path)
    report["vocal_stem"] = str(vocal_path)
    report["elapsed_seconds"] = round(time.time() - t_all, 1)

    rpath = out_path.with_suffix(".report.json")
    rpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成 {report['elapsed_seconds']}s → {out_path}")
    return report


def _place(vocal, offset_seconds, total_samples, sr):
    """把人声按 offset 放到时间线上。offset>0 延后，<0 提前（切掉开头）。"""
    shift = int(round(offset_seconds * sr))
    out = np.zeros(total_samples, dtype=np.float32)
    if shift >= 0:
        src = vocal[: max(0, total_samples - shift)]
        out[shift: shift + len(src)] = src
    else:
        s = min(-shift, len(vocal))
        src = vocal[s:]
        out[: min(len(src), total_samples)] = src[:total_samples]
    return out
