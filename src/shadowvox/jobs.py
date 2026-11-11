"""Shared desktop parameters, versioned export, and serial batch processing."""
import json
import re
import shutil
import tempfile
from pathlib import Path

from .signalsmith import ENGINE

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.aiff', '.wma'}
DESKTOP_RENDERER = ENGINE
DEFAULT_PARAMS = dict(correction=50, beauty=50, fusion=50, ratio=0, mixdown=100,
                      mix_effects=True, dry_has_backing=False)


def options_from(p):
    """感知参数映射，兼容旧预设的修正、音色、空间与人声增益。"""
    from .mastering import TARGET_LUFS, TRUE_PEAK_CEILING
    from .pipeline import Options
    detail, rate, harmonics, density = 0., .10, 1., 1.
    mute_gaps = False
    if 'correction' in p:
        correction, beauty, fusion, mixdown = (p[k]/100 for k in ('correction','beauty','fusion','mixdown'))
        cleanup, tone, space = (min(1.,2*x) for x in (correction,beauty,fusion))
        if not any(k in p for k in ('strength','rhythm_strength','rhythm','denoise_strength','denoise','declip')):
            detail = max(0.,2*correction-1)
            rate += .025*detail
        if not any(k in p for k in ('mix_lowcut_hz','mix_eq','mix_compression','mix_deess')):
            harmonics = tone
        density = mixdown
        p = {
            'strength':min(150*correction,50+50*correction),
            'rhythm':correction > 0., 'rhythm_strength':100*cleanup,
            'denoise':correction > 0., 'denoise_strength':100*cleanup, 'declip':correction > 0.,
            'mix_lowcut_hz':75*tone, 'mix_eq':200*beauty, 'mix_compression':100*tone, 'mix_deess':100*tone,
            'mix_reverb':200*fusion, 'mix_delay':100*space, 'mix_duck':min(140.,200*fusion),
            'vocal_gain_db':.06*p['ratio'],
            **p,
        }
        mute_gaps = bool(p['strength'] or (p['rhythm'] and p['rhythm_strength'])
                         or (p['denoise'] and p['denoise_strength']) or p['declip'])
    return Options(strength=p['strength']/100, pitch_mode='register', max_shift_cents=450, register_detail=detail,
        octave='auto', rhythm=bool(p['rhythm']), rhythm_strength=p['rhythm_strength']/100,
        mix=True, mix_effects=bool(p['mix_effects']), comparison=True, renderer=DESKTOP_RENDERER,
        rhythm_mode='full', rhythm_max_rate=rate, denoise=bool(p['denoise']), denoise_strength=p['denoise_strength']/100,
        declip=bool(p['declip']), vocal_gain_db=float(p['vocal_gain_db']), mix_space=1.,
        mix_lowcut_hz=float(p['mix_lowcut_hz']), mix_eq=p['mix_eq']/100,
        mix_compression=p['mix_compression']/100, mix_deess=p['mix_deess']/100, mix_harmonics=harmonics,
        mix_reverb=p['mix_reverb']/100, mix_delay=p['mix_delay']/100, mix_duck=p['mix_duck']/100,
        fixed_master=True, target_lufs=TARGET_LUFS, true_peak_ceiling=TRUE_PEAK_CEILING, master_density=density,
        dry_has_backing=bool(p.get('dry_has_backing')), mute_reference_gaps=mute_gaps)


def next_version(folder, stem):
    pattern = re.compile(re.escape(stem)+r'_V(\d+)\.wav$', re.I)
    taken = [int(m.group(1)) for f in Path(folder).glob('*.wav') if (m := pattern.match(f.name))]
    return max(taken, default=0)+1


def export_job(dry, song, output, options, session, *, work=None, log=print, audition=False):
    from .pipeline import run
    from .reporting import build, html_report

    capture = {}
    temp = Path(tempfile.mkdtemp(dir=session))/'成品.wav'
    report = run(dry, song, str(temp), work_dir=work, opts=options, log=log, capture=capture)
    result = {'report': report}
    if audition:
        from .preview import prepare
        log('[试听] 准备前后对照')
        tracks, sr, stats = prepare(report['comparison_before'], report['output'])
        result.update(tracks=tracks, sr=sr, preview=stats, capture=capture)
    log('[报告] 测量成品')
    metrics = build(report, report['comparison_before'], report['output'], capture=capture)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(dry).stem
    while True:
        n = next_version(output, stem)
        target = output/f'{stem}_V{n}.wav'
        try:
            stream = target.open('xb')
            break
        except FileExistsError:
            continue
    json_path, html_path = (target.with_suffix(suffix) for suffix in ('.report.json', '.report.html'))
    try:
        with stream, temp.open('rb') as source:
            shutil.copyfileobj(source, stream)
        saved = dict(report, output=str(target), audio_analysis=metrics)
        saved.pop('vocal_stem', None)
        saved.pop('comparison_before', None)
        json_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
        html_path.write_text(html_report(saved, metrics, target), encoding='utf-8')
    except Exception:
        for path in (target, json_path, html_path):
            path.unlink(missing_ok=True)
        raise
    return dict(result, analysis=metrics, version=n, path=str(target), html=str(html_path))


def run_batch(items, output, params, *, work=None, log=print, update=lambda _: None, cancelled=lambda: False):
    """Each item owns its reference and dry-cleanup flag. Failures are isolated."""
    results = []

    def checkpoint(message):
        if cancelled():
            raise InterruptedError('已停止批处理')
        log(message)

    for index, item in enumerate(items):
        if cancelled():
            break
        status = dict(index=index, state='running')
        update(status)
        try:
            dry, song = (Path(item[key]) for key in ('dry', 'song'))
            for path in (dry, song):
                if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                    raise ValueError(f'音频文件不可用：{path}')
            options = options_from(dict(params, dry_has_backing=item.get('dry_has_backing', params.get('dry_has_backing', False))))
            with tempfile.TemporaryDirectory(prefix='shadowvox-batch-') as session:
                result = export_job(dry, song, output, options, session, work=work, log=checkpoint)
            status = dict(index=index, state='done', output=result['path'], html=result['html'])
        except InterruptedError:
            status = dict(index=index, state='cancelled')
        except Exception as exc:
            status = dict(index=index, state='error', error=str(exc))
        results.append(status)
        update(status)
        if status['state'] == 'cancelled':
            break
    return results
