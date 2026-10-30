"""成品报告：从实际音频测量，供桌面界面与离线 HTML 共用。"""
from html import escape
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, report_analysis


CENTERS = (40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000)


def measure(path):
    """FFmpeg 对文件本身测量综合响度、响度范围与真峰值。"""
    info = sf.info(path)
    proc = subprocess.run([audio._ffmpeg_exe(), '-v', 'info', '-nostats', '-i', str(path),
        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
        capture_output=True, check=True, **audio.subprocess_options())
    log = proc.stderr.decode(errors='replace')
    stats = json.loads(log[log.rfind('{'):log.rfind('}') + 1])

    def finite(key):
        value = float(stats[key])
        return round(value, 2) if math.isfinite(value) else None

    return dict(seconds=round(info.duration, 2), sample_rate=info.samplerate,
                channels=info.channels, lufs=finite('input_i'),
                true_peak_dbtp=finite('input_tp'), lra_lu=finite('input_lra'))


def spectrum(path):
    """整曲分块平均频谱，前后同一计算口径；不是效果器传递函数。"""
    size, hop = 4096, 2048
    window = np.hanning(size)
    power = np.zeros(size // 2 + 1, dtype=np.float64)
    count = 0
    with sf.SoundFile(path) as stream:
        sample_rate = stream.samplerate
        tail = np.empty((0, stream.channels), dtype=np.float32)
        while True:
            chunk = stream.read(65536, dtype='float32', always_2d=True)
            if not len(chunk):
                break
            samples = np.concatenate((tail, chunk), axis=0)
            if len(samples) < size:
                tail = samples
                continue
            frames = np.lib.stride_tricks.sliding_window_view(samples, size, axis=0)[::hop]
            transformed = np.fft.rfft(frames * window, axis=2)
            power += np.sum(np.mean(np.abs(transformed) ** 2, axis=1), axis=0)
            count += len(frames)
            tail = samples[len(frames) * hop:]
    if not count:
        return [None] * len(CENTERS)
    power /= count * np.sum(window) ** 2
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    values = []
    for hz in CENTERS:
        mask = (frequencies >= hz / 2**(1/6)) & (frequencies < hz * 2**(1/6))
        if not mask.any():
            mask[np.argmin(abs(frequencies - hz))] = True
        energy = float(np.sum(power[mask]))
        values.append(round(10 * math.log10(max(energy, 1e-14)), 1))
    return values


def build(report, before_path, output_path, capture=None):
    metrics = dict(before=measure(before_path), output=measure(output_path),
                spectrum=dict(centers=list(CENTERS), before=spectrum(before_path),
                              output=spectrum(output_path)),
                source_note='处理前为原始干声与同一分离伴奏的混合，已补偿整体位置；不是原唱成品。')
    options = report.get('options', {})
    metrics['visual'] = dict(schema=1, title=Path(report.get('song', output_path)).stem,
        **report_analysis.results(report, capture),
        mix=dict(before=report_analysis.stereo_scope(before_path), after=report_analysis.stereo_scope(output_path)),
        processing=dict(pitch=options.get('strength', 0) > 0,
                        rhythm=bool(options.get('rhythm') and options.get('rhythm_strength', 1)),
                        mix=bool(options.get('mix_effects'))),
        targets=dict(lufs=options.get('target_lufs'), peak=options.get('true_peak_ceiling')))
    return metrics


def visual_payload(report, metrics, version, output_path):
    """应用和离线报告共用同一模型，不附带歌曲装饰贴图。"""
    return dict(version=version, name=Path(output_path).name, path=str(output_path),
        before=metrics['before'], output=metrics['output'], spectrum=metrics['spectrum'],
        visual={key: value for key, value in metrics.get('visual', {}).items() if key != 'art'},
        summary=summary_rows(report, metrics), parameters=parameter_rows(report),
        note=metrics['source_note'])


def _value(value, unit='', digits=1):
    if value is None:
        return '未测得'
    return f'{value:.{digits}f}{unit}'


def summary_rows(report, metrics):
    options = report.get('options', {})
    rhythm = report.get('rhythm', {})
    align = report.get('align', {})
    denoise = report.get('denoise', {})
    declip = report.get('declip', {})
    if not options.get('rhythm') or not options.get('rhythm_strength', 1.):
        rhythm_text = '关闭局部节奏匹配'
    elif rhythm:
        rhythm_text = f"保护 {_value(rhythm.get('protected_seconds'), ' 秒')}；延长 {_value(rhythm.get('extra_duration_seconds', 0), ' 秒')}"
    else:
        rhythm_text = '未得到节奏统计'
    rows = [
        ('程序版本', report.get('app_version', '旧报告未记录')),
        ('修音力度', f"{options.get('strength', 0) * 100:.0f}%"),
        ('平均音高修正量', _value(align.get('mean_abs_shift_cents'), ' 音分')),
        ('参考匹配置信度', _value(align.get('mean_confidence'), '', 2)),
        ('节奏匹配', rhythm_text),
        ('AI 降噪', f"开启 · {denoise.get('strength', 1) * 100:.0f}%" if denoise.get('enabled') else '关闭'),
        ('削波修复', '已应用' if declip.get('applied') else ('关闭' if declip.get('disabled') else '未检测到明显削波')),
        ('处理时间', _value(report.get('elapsed_seconds'), ' 秒')),
    ]
    visual = metrics.get('visual', {})
    pitch, timing = visual.get('pitch', {}), visual.get('timing', {})
    measured = []
    if pitch.get('available'):
        measured += [('实测音高偏差中位数', f"{pitch['before_median']:.1f} → {pitch['after_median']:.1f} 音分"),
                     ('共同有效音高帧', str(pitch['common_frames']))]
    if timing.get('available'):
        measured += [('实测入点偏差中位数', f"{timing['before_median_ms']:.0f} → {timing['after_median_ms']:.0f} ms"),
                     ('共同原唱入点', str(timing['paired_onsets']))]
    rows = measured+rows
    intervals = rhythm.get('protected_intervals_seconds', [])
    if intervals:
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1] + .05:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        def clock(seconds):
            minutes, tenths = divmod(max(0, round(seconds * 10)), 600)
            return f'{minutes:02d}:{tenths/10:04.1f}'
        positions = '、'.join(f'{clock(start)}–{clock(end)}' for start, end in merged[:6])
        if len(merged) > 6:
            positions += f' 等 {len(merged)} 段'
        rows.insert(4, ('建议试听位置', positions))
    return rows


def metric_rows(metrics):
    before, out = metrics['before'], metrics['output']
    return [
        ('综合响度', _value(before.get('lufs'), ' LUFS'), _value(out.get('lufs'), ' LUFS')),
        ('真峰值', _value(before.get('true_peak_dbtp'), ' dBTP'), _value(out.get('true_peak_dbtp'), ' dBTP')),
        ('响度范围', _value(before.get('lra_lu'), ' LU'), _value(out.get('lra_lu'), ' LU')),
        ('时长', _value(before.get('seconds'), ' 秒'), _value(out.get('seconds'), ' 秒')),
    ]


PARAM_LABELS = {
    'rhythm_strength': ('节奏强度', '%', 100), 'denoise_strength': ('降噪强度', '%', 100),
    'vocal_gain_db': ('人声音量', ' dB', 1), 'mix_lowcut_hz': ('低切', ' Hz', 1),
    'mix_eq': ('均衡', '%', 100), 'mix_compression': ('压缩', '%', 100),
    'mix_deess': ('去齿', '%', 100), 'mix_reverb': ('混响', '%', 100),
    'mix_delay': ('延时', '%', 100), 'mix_duck': ('伴奏避让', '%', 100),
    'target_lufs': ('目标响度', ' LUFS', 1), 'true_peak_ceiling': ('真峰值上限', ' dBTP', 1),
}


def parameter_rows(report):
    options = report.get('options', {})
    rows = []
    for key, (label, unit, scale) in PARAM_LABELS.items():
        if key in options:
            rows.append((label, _value(options[key] * scale, unit)))
    rows.append(('混音效果', '开启' if options.get('mix_effects') else '关闭'))
    return rows


def svg_spectrum(data):
    centers = data['centers']
    values = [x for name in ('before', 'output') for x in data[name] if x is not None]
    if not values:
        return '<p>频谱数据不可用。</p>'
    low = math.floor(min(values) / 10) * 10
    high = math.ceil(max(values) / 10) * 10
    if high <= low:
        high = low + 10
    def points(name):
        result = []
        for hz, db in zip(centers, data[name]):
            if db is not None:
                x = 56 + 600 * math.log(hz / centers[0]) / math.log(centers[-1] / centers[0])
                y = 20 + 180 * (high - db) / (high - low)
                result.append(f'{x:.1f},{y:.1f}')
        return ' '.join(result)
    lines = ''.join(f'<line x1="56" x2="656" y1="{20 + 180*i/4}" y2="{20 + 180*i/4}" stroke="#efd5df"/>'
                    for i in range(5))
    labels = ''.join(f'<text x="48" y="{24 + 180*i/4}" text-anchor="end">{high - (high-low)*i/4:.0f}</text>'
                     for i in range(5))
    return (f'<svg viewBox="0 0 680 235" role="img" aria-label="处理前后平均频谱能量对比">'
            f'<g fill="#62505a" font-size="12">{lines}{labels}'
            '<text x="56" y="221">40 Hz</text><text x="630" y="221">16 kHz</text></g>'
            f'<polyline points="{points("before")}" fill="none" stroke="#817980" stroke-width="2"/>'
            f'<polyline points="{points("output")}" fill="none" stroke="#b32965" stroke-width="3"/></svg>')


def html_report(report, metrics, output_path):
    def table(rows):
        return ''.join('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows)
    filename = escape(Path(output_path).name)
    model = visual_payload(report, metrics, None, output_path)
    model.pop('path', None)  # 离线分享文件不需要本机完整路径。
    serialized = json.dumps(model, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e')
    ui = Path(__file__).with_name('ui')
    css = (ui/'report.css').read_text(encoding='utf-8')
    script = (ui/'report.js').read_text(encoding='utf-8')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHADOWVOX 声音改造战报 · {filename}</title>
<style>{css}</style><body class="report-document"><main class="report-shell">
<header class="rp-toolbar"><span class="rp-title">{filename}</span>
<button id="export-png" class="rp-action" type="button">导出分享图</button></header>
<div class="rp-poster" id="report-poster"></div><p class="rp-feedback" id="report-status" role="status"></p>
<details class="rp-details"><summary>技术详情与测量口径</summary>
<div class="rp-detail-columns"><section><h2>处理记录</h2><table>{table(summary_rows(report, metrics))}</table></section>
<section><h2>处理参数</h2><table>{table(parameter_rows(report))}</table></section></div>
<p>{escape(metrics['source_note'])}</p>
<p>音高按共同有效帧比较音级偏差，保留演唱音区；节奏只比较匹配至同一原唱入点的前后结果。
音色比较原始干声与空间效果前的汇总人声，按活动段 RMS 电平归一；频段数值是全链路的相对能量变化，
不代表清晰度或空气感评分，也不是单项 EQ 的增益。声场取真实成品。</p></details>
</main><script>{script}</script><script>
const reportData = {serialized};
document.getElementById('report-poster').innerHTML = ShadowReport.svg(reportData);
document.getElementById('export-png').addEventListener('click', async event => {{
 const button = event.currentTarget, status = document.getElementById('report-status'); button.disabled = true;
 try {{ const url = await ShadowReport.png(reportData); const a = document.createElement('a');
 a.href = url; a.download = reportData.name.replace(/\\.wav$/i, '')+'.report.png'; a.click();
 status.textContent = '已发起 PNG 下载（1080×1440）'; }}
 catch (error) {{ status.textContent = '导出失败：'+error.message; }} finally {{ button.disabled = false; }}
}});
</script></body></html>'''
