"""前端桥接：文件选择、处理任务、草稿试听、播放器、版本输出与设置。

界面状态由前端持有；这里只做真实处理与音频，并把结果推给前端。
"""
import hashlib
import base64
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFileDialog

from . import __version__, config as C
from .player import ComparisonPlayer

AUDIO_FILTER = '音频文件 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.aiff *.wma);;所有文件 (*)'
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.aiff', '.wma'}
# 日志前缀 → (阶段序号, 阶段名)；与前端主按钮内的 6 个阶段刻度一一对应。
STAGES = {'[1/6]': (0, '读取音频'), '[修复]': (0, '修复削波'), '[降噪]': (0, 'AI 降噪'),
          '[2/6]': (1, '分离原曲'), '[3/6]': (2, '分析音高'), '[节奏]': (2, '整句对齐节奏'),
          '[4/6]': (3, '匹配音高'), '[5/6]': (4, '修正人声'), '[6/6]': (5, '合成成品'),
          '[混音]': (5, '混音处理'), '[试听]': (5, '准备对比试听'), '[报告]': (5, '测量成品')}
WAVE_BINS = 1600


def options_from(p):
    """前端参数（自然单位）→ 处理选项；默认值与 Options 及 v7 已确认的混音一致。"""
    from .pipeline import Options
    return Options(strength=p['strength']/100, pitch_mode='register', max_shift_cents=450,
        octave='auto', rhythm=bool(p['rhythm']), rhythm_strength=p['rhythm_strength']/100,
        mix=True, mix_effects=bool(p['mix_effects']), comparison=True, renderer='psola',
        rhythm_mode='full', denoise=bool(p['denoise']), denoise_strength=p['denoise_strength']/100,
        declip=bool(p['declip']), vocal_gain_db=float(p['vocal_gain_db']), mix_space=1.,
        mix_lowcut_hz=float(p['mix_lowcut_hz']), mix_eq=p['mix_eq']/100,
        mix_compression=p['mix_compression']/100, mix_deess=p['mix_deess']/100,
        mix_reverb=p['mix_reverb']/100, mix_delay=p['mix_delay']/100, mix_duck=p['mix_duck']/100,
        target_lufs=float(p['target_lufs']), true_peak_ceiling=float(p['true_peak_ceiling']))


def envelope(track, bins=WAVE_BINS):
    level = np.abs(track).max(axis=1)
    size = len(level)//bins*bins
    blocks = level[:size].reshape(bins, -1)
    return blocks.max(axis=1), np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1))


def report_payload(report, metrics, version, path):
    from .reporting import visual_payload
    return visual_payload(report, metrics, version, path)


def next_version(folder, stem):
    pattern = re.compile(re.escape(stem)+r'_V(\d+)\.wav$', re.I)
    taken = [int(m.group(1)) for f in Path(folder).glob('*.wav') if (m := pattern.match(f.name))]
    return max(taken, default=0)+1


class Worker(QThread):
    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, dry, song, output_dir, options, session, work=None):
        super().__init__()
        self.dry, self.song, self.output_dir, self.options = dry, song, Path(output_dir), options
        self.session, self.work, self.error_detail = session, work, ''

    def run(self):
        try:
            from .pipeline import run
            from .preview import prepare
            from .reporting import build, html_report
            def log(message):
                if self.isInterruptionRequested():
                    raise InterruptedError('已取消处理，可以调整设置后重试。')
                self.message.emit(message)
            capture = {}
            temp = Path(tempfile.mkdtemp(dir=self.session))/'成品.wav'
            report = run(self.dry, self.song, str(temp), work_dir=self.work,
                         opts=self.options, log=log, capture=capture)
            log('[试听] 准备前后对照')
            tracks, sr, stats = prepare(report['comparison_before'], report['output'])
            log('[报告] 测量成品')
            metrics = build(report, report['comparison_before'], report['output'], capture=capture)
            # 输出目录按版本自动保存成品与报告，不覆盖已有版本。
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(self.dry).stem
            n = next_version(self.output_dir, stem)
            target = self.output_dir/f'{stem}_V{n}.wav'
            shutil.copyfile(temp, target)
            saved = dict(report); saved['output'] = str(target); saved['audio_analysis'] = metrics
            saved.pop('vocal_stem', None); saved.pop('comparison_before', None)
            target.with_suffix('.report.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
            target.with_suffix('.report.html').write_text(html_report(saved, metrics, target), encoding='utf-8')
            log('[报告] 测量完成')
            self.succeeded.emit(dict(report=report, tracks=tracks, sr=sr, preview=stats, capture=capture,
                                     analysis=metrics, version=n, path=str(target)))
        except InterruptedError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            import traceback
            self.error_detail = traceback.format_exc()
            self.failed.emit(str(exc) if str(exc).startswith(('AI 降噪失败', '未找到 AI 降噪组件')) else
                             '处理未完成。请确认两个文件可正常播放，并检查输出目录权限与磁盘空间后重试。')


class DraftWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, cache, options, position, revision):
        super().__init__()
        self.cache, self.options, self.position, self.revision = cache, options, position, revision

    def run(self):
        try:
            start, samples = self.cache.render(self.options, self.position)
            if not self.isInterruptionRequested():
                self.succeeded.emit((self.revision, start, samples))
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class VersionLoader(QThread):
    """切换成品版本：重新准备同伴奏等响度的前后对照。"""
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, version, before, after):
        super().__init__()
        self.version, self.before, self.after = version, before, after

    def run(self):
        try:
            from .preview import prepare
            tracks, sr, stats = prepare(self.before, self.after)
            self.succeeded.emit((self.version, tracks, sr))
        except Exception as exc:
            self.failed.emit(str(exc))


class Bridge(QObject):
    stage = Signal(int, float, str)       # 阶段序号、阶段内进度（<0 未知）、阶段名
    finished = Signal(str)                # JSON：新版本与报告
    failed = Signal(str)                  # JSON：message / detail
    playerState = Signal(str)             # JSON：位置、时长、播放、电平、当前版本
    waveform = Signal(str)                # JSON：三版本包络；空表示清空
    waveformSegment = Signal(str)         # JSON：草稿重算的包络区间
    draftState = Signal(str)              # JSON：preparing / ready / error
    versionLoaded = Signal(str)           # JSON：切换后的报告
    filesDropped = Signal(str)            # JSON：拖入路径与落点
    vstStatus = Signal(str)               # JSON：插件状态文字与是否忙碌

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = ComparisonPlayer(); self.player.volume = 1.
        self.session = tempfile.TemporaryDirectory(prefix='shadowvox-preview-')
        self.inputs = {'song': '', 'dry': ''}
        self.worker = self.draft_worker = self.loader = self.plugin_process = None
        self.work_override = None; self.preview_cache = None; self.params = None
        self.versions = {}; self.current = None
        self.draft_revision = 0; self.draft_ready_revision = -1; self.draft_range = (0, 0)
        self.draft_failed = False; self.last_state = None
        self.draft_timer = QTimer(self); self.draft_timer.setSingleShot(True)
        self.draft_timer.timeout.connect(self.start_draft)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)

    # ── 文件 ──
    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    @Slot(str, result=str)
    def pickAudio(self, kind):
        title = '选择原曲' if kind == 'song' else '选择演唱干音'
        value, _ = QFileDialog.getOpenFileName(None, title, str(Path(self.inputs[kind]).parent) if self.inputs[kind] else '', AUDIO_FILTER)
        return self.setInput(kind, value) if value else ''

    @Slot(str, str, result=str)
    def setInput(self, kind, path):
        if self.busy() or kind not in self.inputs or not Path(path).is_file():
            return ''
        self.inputs[kind] = str(Path(path)); self.invalidate()
        return json.dumps(dict(path=self.inputs[kind], name=Path(path).name), ensure_ascii=False)

    @Slot(str, result=str)
    def pickOutput(self, current):
        return QFileDialog.getExistingDirectory(None, '选择输出目录', current or '') or ''

    @Slot(str, result=bool)
    def folderExists(self, path):
        return bool(path) and Path(path).is_dir()

    @Slot(str)
    def openFolder(self, path):
        import subprocess
        subprocess.Popen(['explorer', '/select,', str(Path(path).resolve())])

    @Slot(str, result=bool)
    def copyText(self, text):
        QApplication.clipboard().setText(text)
        return QApplication.clipboard().text() == text

    @Slot(str, int, result=str)
    def saveReportImage(self, data_url, version):
        if version not in self.versions:
            return json.dumps({'error': '此成品版本不可用。'}, ensure_ascii=False)
        target = Path(self.versions[version]['audio']).with_suffix('.report.png')
        filename, _ = QFileDialog.getSaveFileName(None, '保存分享图', str(target), 'PNG 图片 (*.png)')
        if not filename:
            return json.dumps({'cancelled': True})
        try:
            if not data_url.startswith('data:image/png;base64,'):
                raise ValueError('图片格式错误')
            image = QImage.fromData(base64.b64decode(data_url.split(',', 1)[1]), 'PNG')
            if image.isNull() or image.width() != 1080 or image.height() != 1440:
                raise ValueError('分享图尺寸错误')
            if not image.save(filename, 'PNG'):
                raise OSError('无法写入图片文件')
            return json.dumps({'saved': filename}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({'error': str(exc)}, ensure_ascii=False)

    @Slot(result=str)
    def appVersion(self):
        return __version__

    def invalidate(self):
        self.draft_revision += 1; self.draft_timer.stop()
        if self.draft_worker and self.draft_worker.isRunning(): self.draft_worker.requestInterruption()
        self.preview_cache = None; self.player.clear(); self.versions = {}; self.current = None
        self.waveform.emit('{}'); self.last_state = None

    # ── 处理 ──
    @Slot('QVariantMap', result=bool)
    def process(self, params):
        if self.busy() or not all(Path(p).is_file() for p in self.inputs.values()) or not params.get('output'):
            return False
        self.params = dict(params)
        self.worker = Worker(self.inputs['dry'], self.inputs['song'], params['output'],
                             options_from(params), self.session.name, self.work_override)
        self.worker.message.connect(self.on_message); self.worker.succeeded.connect(self.on_success)
        self.worker.failed.connect(self.on_failure)
        self.started = time.monotonic(); self.worker.start()
        return True

    @Slot()
    def cancel(self):
        if self.busy(): self.worker.requestInterruption()

    # 线程回调必须是静态声明的 Slot：运行中新增的动态槽会改变 QWebChannel 已发布的方法序号，前端将收不到任何信号。
    @Slot(str)
    def on_message(self, message):
        if self.worker and self.worker.isInterruptionRequested(): return
        if message.startswith('[合成]'):
            try: self.stage.emit(4, float(message.split()[-1].rstrip('%'))/100, '修正人声')
            except ValueError: pass
            return
        for prefix, (index, label) in STAGES.items():
            if message.startswith(prefix):
                self.stage.emit(index, -1., label); break

    @Slot(str)
    def on_failure(self, message):
        self.failed.emit(json.dumps(dict(message=message, detail=self.worker.error_detail), ensure_ascii=False))

    @Slot(object)
    def on_success(self, result):
        n = result['version']
        self.versions[n] = dict(before=result['report']['comparison_before'], audio=result['report']['output'],
                                report=report_payload(result['report'], result['analysis'], n, result['path']))
        from .live_preview import PreviewCache
        self.preview_cache = PreviewCache(result['capture'], result['report'], result['preview'])
        self.load_tracks(n, result['tracks'], result['sr'], select=2)
        rhythm = result['report'].get('rhythm', {})
        payload = dict(report=self.versions[n]['report'],
                       versions=[dict(n=k, name=v['report']['name'], path=v['report']['path']) for k, v in sorted(self.versions.items())],
                       protected=rhythm.get('protected_seconds', 0) > .5)
        self.finished.emit(json.dumps(payload, ensure_ascii=False))

    def load_tracks(self, n, tracks, sr, select=None):
        before, after = tracks
        playing, position = self.player.playing, self.player.position
        loop, delta, selected = self.player.loop, self.player.delta, self.player.selected
        self.player.load([before, after.copy(), after], sr)
        self.player.position = min(position, self.player.frames)
        self.player.loop, self.player.delta = loop, delta
        self.player.select(selected if select is None else select)
        self.current = n; self.draft_range = (0, self.player.frames)
        self.draft_ready_revision = self.draft_revision; self.draft_failed = False
        envelopes = [envelope(t) for t in self.player.tracks]
        scale = max(float(np.percentile(peak, 99.5)) for peak, _ in envelopes) or 1.
        self.wave_scale = scale
        self.waveform.emit(json.dumps(dict(duration=self.player.frames/sr, tracks=[
            dict(peak=np.round(np.minimum(p/scale, 1), 3).tolist(), rms=np.round(np.minimum(r/scale, 1), 3).tolist())
            for p, r in envelopes])))
        if playing:
            try: self.player.play()
            except Exception: pass

    @Slot(int)
    def selectVersion(self, n):
        if n not in self.versions or n == self.current or (self.loader and self.loader.isRunning()): return
        v = self.versions[n]
        self.loader = VersionLoader(n, v['before'], v['audio'])
        self.loader.succeeded.connect(self.on_version_loaded)
        self.loader.failed.connect(self.on_version_failed)
        self.loader.start()

    @Slot(str)
    def on_version_failed(self, detail):
        self.failed.emit(json.dumps(dict(message='无法载入该版本试听。', detail=detail), ensure_ascii=False))

    @Slot(object)
    def on_version_loaded(self, result):
        n, tracks, sr = result
        self.load_tracks(n, tracks, sr)
        self.versionLoaded.emit(json.dumps(self.versions[n]['report'], ensure_ascii=False))

    # ── 草稿 ──
    @Slot('QVariantMap')
    def draft(self, params):
        self.params = dict(params)
        if self.preview_cache is None or self.busy(): return
        self.draft_revision += 1; self.draft_failed = False
        self.player.select(1)
        if self.draft_worker and self.draft_worker.isRunning(): self.draft_worker.requestInterruption()
        self.draftState.emit(json.dumps(dict(state='preparing')))
        self.draft_timer.start(200)

    def draft_due(self):
        return (self.player.selected == 1 and self.preview_cache is not None and not self.draft_failed and
                (self.draft_ready_revision != self.draft_revision or self.player.position < self.draft_range[0] or
                 (self.player.position >= self.draft_range[1]-2*self.player.sr and self.draft_range[1] < self.player.frames)))

    @Slot()
    def start_draft(self):
        if self.preview_cache is None or self.busy() or self.params is None: return
        if self.draft_worker and self.draft_worker.isRunning(): return
        self.draftState.emit(json.dumps(dict(state='preparing')))
        worker = DraftWorker(self.preview_cache, options_from(self.params), self.player.position/self.player.sr, self.draft_revision)
        self.draft_worker = worker
        worker.succeeded.connect(self.on_draft_success); worker.failed.connect(self.on_draft_failure)
        worker.finished.connect(self.on_draft_finished); worker.start()

    @Slot(object)
    def on_draft_success(self, result):
        revision, start, samples = result
        if revision != self.draft_revision or self.preview_cache is None: return
        self.player.replace_segment(1, start, samples)
        end = start+len(samples); self.draft_ready_revision = revision; self.draft_range = (start, end)
        block = self.player.frames//WAVE_BINS
        if block:
            first, last = start//block, min(WAVE_BINS, (end+block-1)//block)
            peak, rms = envelope(self.player.tracks[1][first*block:last*block], last-first)
            s = self.wave_scale
            self.waveformSegment.emit(json.dumps(dict(index=1, first=int(first),
                peak=np.round(np.minimum(peak/s, 1), 3).tolist(), rms=np.round(np.minimum(rms/s, 1), 3).tolist())))
        sr = self.player.sr
        self.draftState.emit(json.dumps(dict(state='ready', start=start/sr, end=end/sr)))

    @Slot(str)
    def on_draft_failure(self, detail):
        self.draft_failed = True
        self.draftState.emit(json.dumps(dict(state='error', detail=detail), ensure_ascii=False))

    @Slot()
    def on_draft_finished(self):
        self.draft_worker = None
        if self.preview_cache is not None and not self.busy() and self.draft_due(): self.draft_timer.start(50)

    # ── 播放 ──
    @Slot()
    def toggle(self):
        if not self.player.tracks: return
        if self.player.playing:
            self.player.pause()
        else:
            try: self.player.play()
            except Exception:
                self.failed.emit(json.dumps(dict(message='无法播放，请检查系统音频输出设备；成品已保存在输出目录。', detail=''), ensure_ascii=False))

    @Slot(float)
    def seek(self, seconds):
        self.player.seek(seconds)
        if self.player.selected == 1 and self.preview_cache is not None: self.draft_timer.start(150)

    @Slot(int)
    def select(self, index):
        self.player.select(index); self.player.delta = False   # 切换音源时关闭差值监听
        if self.draft_due() and not self.draft_timer.isActive(): self.draft_timer.start(50)

    @Slot(float)
    def setVolume(self, value):
        self.player.volume = max(0., min(1., value))

    @Slot(bool, float, float)
    def setLoop(self, enabled, start, end):
        if enabled: self.player.set_loop(start, end)
        else: self.player.loop = None

    @Slot(bool)
    def setDelta(self, enabled):
        self.player.delta = enabled and self.player.selected != 0

    @Slot()
    def tick(self):
        p = self.player
        if not p.tracks: return
        state = dict(pos=round(p.position/p.sr, 3), dur=round(p.frames/p.sr, 3), playing=p.playing,
                     sel=p.selected, delta=p.delta,
                     levels=[round(float(x), 4) for x in (p.levels if p.playing else (0, 0))])
        if state != self.last_state:
            self.last_state = state; self.playerState.emit(json.dumps(state))
        if self.draft_due() and not self.draft_timer.isActive() and not (self.draft_worker and self.draft_worker.isRunning()):
            self.draft_timer.start(50)

    # ── 设置：缓存与 VST3 ──
    @Slot(result=str)
    def cacheInfo(self):
        root = Path(self.work_override or C.WORK_DIR)
        size = sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) if root.is_dir() else 0
        return json.dumps(dict(bytes=size, busy=self.busy()))

    @Slot(result=str)
    def clearCache(self):
        if not self.busy():
            shutil.rmtree(Path(self.work_override or C.WORK_DIR), ignore_errors=True)
        return self.cacheInfo()

    def plugin_index(self):
        return C.DATA_ROOT/'settings'/'vst3'/'plugins.json'

    def plugin_paths(self):
        index = self.plugin_index()
        return json.loads(index.read_text(encoding='utf-8')) if index.is_file() else []

    @Slot(result=str)
    def vstList(self):
        return json.dumps([dict(name=Path(p).stem, path=p) for p in self.plugin_paths()], ensure_ascii=False)

    @Slot(str, result=str)
    def vstAdd(self, kind):
        path = (QFileDialog.getOpenFileName(None, '选择 VST3 插件', '', 'VST3 插件 (*.vst3)')[0] if kind == 'file'
                else QFileDialog.getExistingDirectory(None, '选择 VST3 插件文件夹'))
        if path:
            path = str(Path(path).resolve())
            if Path(path).suffix.lower() != '.vst3':
                self.vst_status('请选择 VST3 文件或以 .vst3 结尾的文件夹。')
            else:
                paths = self.plugin_paths()
                if os.path.normcase(path) not in [os.path.normcase(p) for p in paths]:
                    paths.append(path); self.plugin_index().parent.mkdir(parents=True, exist_ok=True)
                    self.plugin_index().write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding='utf-8')
                self.vst_status('插件已添加，打开界面即可调整设置。')
        return self.vstList()

    def vst_status(self, text, busy=False):
        self.vstStatus.emit(json.dumps(dict(text=text, busy=busy), ensure_ascii=False))

    @Slot(int)
    def vstOpen(self, index):
        paths = self.plugin_paths()
        if not 0 <= index < len(paths) or (self.plugin_process and self.plugin_process.state() != QProcess.NotRunning):
            return
        path = paths[index]
        if not Path(path).exists():
            self.vst_status('插件文件不存在，请重新添加。'); return
        folder = self.plugin_index().parent; folder.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(os.path.normcase(path).encode('utf-8')).hexdigest()[:20]
        error = folder/(key+'.error.txt'); error.unlink(missing_ok=True)
        process = QProcess(self); self.plugin_process = process
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYINSTALLER_RESET_ENVIRONMENT', '1'); process.setProcessEnvironment(environment)
        def finished(code, status):
            if code == 0 and status == QProcess.NormalExit:
                self.vst_status('插件设置已保存；当前不参与音频处理。')
            else:
                detail = error.read_text(encoding='utf-8').strip().splitlines()[-1] if error.is_file() else '插件进程异常退出'
                self.vst_status('无法打开插件：'+detail)
        process.finished.connect(finished)
        process.errorOccurred.connect(lambda e: e == QProcess.FailedToStart and self.vst_status('无法启动插件界面。'))
        args = [] if getattr(sys, 'frozen', False) else [str(C.ROOT/'desktop.py')]
        process.start(sys.executable, args+['--plugin-editor', path, str(folder/(key+'.state')), str(error)])
        self.vst_status('正在打开插件界面；关闭插件界面后自动保存设置。', True)

    # ── 关闭 ──
    def can_close(self):
        if self.busy(): return '请先停止并等待当前计算结束，再关闭窗口。'
        if self.plugin_process and self.plugin_process.state() != QProcess.NotRunning:
            return '请先关闭插件界面，以保存设置。'
        if self.draft_worker and self.draft_worker.isRunning():
            self.draft_worker.requestInterruption()
            if not self.draft_worker.wait(3000): return '正在结束草稿计算，请稍后关闭。'
        return ''

    def shutdown(self):
        self.timer.stop(); self.draft_timer.stop()
        if self.loader and self.loader.isRunning(): self.loader.wait(10000)
        self.player.clear(); self.session.cleanup()
