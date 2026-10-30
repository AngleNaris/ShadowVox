"""ShadowVox 桌面外壳：QWebEngineView 加载 ui/index.html，经 QWebChannel 与 Bridge 通信。"""
import json
import os
import sys
from pathlib import Path

# 与参考工程一致：Windows 边框缩放后 D3D 合成可能显示旧画面，改用 OpenGL 合成。
if sys.platform == 'win32':
    os.environ.setdefault('QSG_RHI_BACKEND', 'opengl')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-sandbox --no-sandbox')
os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')

from PySide6.QtCore import QEvent, QPoint, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow

from . import __version__, config as C
from .gui_bridge import AUDIO_EXTS, Bridge

UI_INDEX = Path(__file__).resolve().parent/'ui'/'index.html'
BASE = '#fffafb'
MIN_WINDOW, DEFAULT_WINDOW = (720, 600), (900, 680)
WINDOW_LAYOUT = 2


class DropView(QWebEngineView):
    """拖入文件：事件落在 WebEngine 内部渲染控件上，需在 focusProxy 上拦截；JS 拿不到完整路径。"""
    dropped = Signal(list, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True); self.installed = False

    def event(self, ev):
        if not self.installed and ev.type() == QEvent.Type.ChildAdded:
            QTimer.singleShot(0, self.install)
        return super().event(ev)

    def install(self):
        proxy = self.focusProxy()
        if proxy is not None and not self.installed:
            proxy.installEventFilter(self); self.installed = True

    def eventFilter(self, obj, ev):
        kind = ev.type()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove) and ev.mimeData().hasUrls():
            ev.acceptProposedAction(); return True
        if kind == QEvent.Type.Drop and ev.mimeData().hasUrls():
            paths = [str(Path(u.toLocalFile())) for u in ev.mimeData().urls() if u.isLocalFile()]
            paths = [p for p in paths if Path(p).is_dir() or Path(p).suffix.lower() in AUDIO_EXTS]
            ev.acceptProposedAction()
            if paths:
                pos = ev.position().toPoint(); self.dropped.emit(paths, pos.x(), pos.y())
            return True
        return super().eventFilter(obj, ev)


class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ShadowVox · 修音工作台')
        self.setWindowIcon(QIcon(str(C.ROOT/'assets/shadowvox.ico')))
        self.setMinimumSize(*MIN_WINDOW); self.restore_geometry()
        palette = self.palette(); palette.setColor(QPalette.Window, QColor(BASE)); self.setPalette(palette)
        self.view = DropView(self); self.view.setContextMenuPolicy(Qt.NoContextMenu)
        # 命名 Profile + 持久存储：前端 localStorage 记住参数、预设与输出目录。
        profile = QWebEngineProfile('ShadowVoxProfile', self.view)
        storage = C.DATA_ROOT/'webview-storage'; storage.mkdir(parents=True, exist_ok=True)
        profile.setPersistentStoragePath(str(storage))
        self.view.setPage(QWebEnginePage(profile, self.view))
        self.view.page().setBackgroundColor(QColor(BASE))
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.bridge = Bridge(self)
        self.channel = QWebChannel(self.view); self.channel.registerObject('bridge', self.bridge)
        self.view.page().setWebChannel(self.channel)
        self.view.dropped.connect(lambda paths, x, y: self.bridge.filesDropped.emit(
            json.dumps(dict(paths=paths, x=x, y=y), ensure_ascii=False)))
        self.setCentralWidget(self.view)
        self.view.load(QUrl.fromLocalFile(str(UI_INDEX)))

    def restore_geometry(self):
        override = os.environ.get('SHADOWVOX_WINDOW', '')
        if 'x' in override:
            w, h = (int(v) for v in override.split('x', 1)); self.resize(w, h); return
        settings = QSettings('ShadowVox', 'ShadowVox')
        size, pos = settings.value('window/size'), settings.value('window/pos')
        screen = self.screen().availableGeometry()
        current_layout = settings.value('window/layout', 0, type=int) == WINDOW_LAYOUT
        if current_layout and isinstance(size, QSize) and isinstance(pos, QPoint) and any(
                s.availableGeometry().contains(pos) for s in QApplication.screens()):
            self.resize(min(size.width(), screen.width()), min(size.height(), screen.height())); self.move(pos); return
        w, h = min(DEFAULT_WINDOW[0], screen.width()-32), min(DEFAULT_WINDOW[1], screen.height()-32)
        self.resize(w, h); self.move(screen.center().x()-w//2, screen.center().y()-h//2)

    def closeEvent(self, event):
        reason = self.bridge.can_close()
        if reason:
            self.bridge.failed.emit(json.dumps(dict(message=reason, detail=''), ensure_ascii=False))
            event.ignore(); return
        settings = QSettings('ShadowVox', 'ShadowVox')
        settings.setValue('window/size', self.size()); settings.setValue('window/pos', self.pos())
        settings.setValue('window/layout', WINDOW_LAYOUT)
        self.bridge.shutdown(); event.accept()


def run_check(window, app, folder):
    """独立 EXE 验收：已有音频试听，以及可选的实际报告渲染和 PNG 导出。"""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    bridge = window.bridge
    report_state = {}

    def write(state):
        p = bridge.player
        window.grab().save(str(folder/'frozen-gui.png'))
        data = dict(app_version=__version__, visible=window.isVisible(), width=window.width(), height=window.height(),
                    playback_enabled=bool(p.tracks), frames_written=p.frames_written, position=p.position,
                    selected=p.selected, stream_active=bool(p.stream and p.stream.active),
                    **report_state, **json.loads(state or '{}'))
        (folder/'frozen-gui.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        window.close(); app.quit()

    def finish():
        window.view.page().runJavaScript('JSON.stringify(window.__svState ? window.__svState() : {})', 0, write)

    def check_report():
        request = folder/'report-request.json'
        if not request.exists():
            finish(); return
        import base64
        import time
        from PySide6.QtGui import QImage
        from .reporting import html_report, visual_payload
        saved = json.loads(Path(json.loads(request.read_text(encoding='utf-8'))['report']).read_text(encoding='utf-8'))
        metrics = saved['audio_analysis']
        (folder/'frozen-report.html').write_text(html_report(saved,metrics,saved['output']),encoding='utf-8')
        payload = json.dumps(visual_payload(saved,metrics,1,saved['output']),ensure_ascii=False)
        script = ('window.__svCheckPng=null; window.__svCheckError=null; renderReport('+payload+', "reveal"); '
                  'switchTab("report"); ShadowReport.png('+payload+').then(url=>window.__svCheckPng=url)'
                  '.catch(error=>window.__svCheckError=error.message); 1')
        window.view.page().runJavaScript(script)
        deadline = time.monotonic()+30

        def received(state):
            data = json.loads(state or '{}')
            if data.get('png'):
                image = QImage.fromData(base64.b64decode(data['png'].split(',',1)[1]))
                if image.width() == 1080 and image.height() == 1440 and image.save(str(folder/'frozen-report.png')):
                    report_state.update(report_png=True, report_summary_rows=data['summary_rows'])
                    finish(); return
                data['error'] = 'PNG 尺寸错误或保存失败'
            if data.get('error') or time.monotonic()>deadline:
                (folder/'gui-check-error.txt').write_text(data.get('error') or '报告导出超时',encoding='utf-8')
                window.close(); app.exit(1); return
            QTimer.singleShot(100,poll)

        def poll():
            window.view.page().runJavaScript(
                'JSON.stringify({png:window.__svCheckPng,error:window.__svCheckError,'
                'summary_rows:document.querySelectorAll("#rp-summary dt").length})',0,received)
        QTimer.singleShot(200,poll)

    def prepare():
        request = folder/'preview-request.json'
        if not request.exists():
            check_report(); return
        try:
            from .preview import prepare as load
            data = json.loads(request.read_text(encoding='utf-8'))
            for key in ('song', 'dry'): bridge.inputs[key] = data[key]
            tracks, sr, _ = load(data['before'], data['after'])
            bridge.load_tracks(0, tracks, sr, select=2)
            bridge.setVolume(.02); bridge.seek(64.); bridge.toggle()
            QTimer.singleShot(250, lambda: bridge.select(0))
            QTimer.singleShot(800, check_report)
        except Exception:
            import traceback
            (folder/'gui-check-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            window.close(); app.exit(1)
    window.view.loadFinished.connect(lambda ok: QTimer.singleShot(1500, prepare))


def main(check_folder=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('ShadowVox')
    app.setApplicationVersion(__version__)
    window = StudioWindow(); window.show()
    if check_folder:
        run_check(window, app, check_folder)
    return app.exec()
