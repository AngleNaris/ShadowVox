"""WebView 工作台验收：真实 Qt 键鼠事件、布局宽度、预设、整曲处理、版本保存、草稿与报告。

需要可显示的桌面会话（QtWebEngine 不支持 offscreen）。证据截图写到 out/desktop_preview/shadowvox_ui。
"""
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import os
os.environ['SHADOWVOX_WINDOW'] = '980x740'
from PySide6.QtCore import QEvent, QEventLoop, QPoint, QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from shadowvox.gui import StudioWindow
from shadowvox import __version__, config as C

OUT = ROOT/'out/releases'/__version__/'workflow'


def contrast_checks():
    css = (ROOT/'src/shadowvox/ui/style.css').read_text(encoding='utf-8')
    root = css.split(':root {', 1)[1].split('}', 1)[0]
    tokens = dict(re.findall(r'--([a-z-]+):\s*(#[0-9a-f]{6})', root))
    def lum(value):
        rgb = [int(value[i:i+2], 16)/255 for i in (1, 3, 5)]
        rgb = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(a*b for a, b in zip(rgb, (.2126, .7152, .0722)))
    result = {}
    for fg, bg, minimum in [('text', 'base', 4.5), ('muted', 'base', 4.5), ('muted', 'surface', 4.5), ('ink', 'accent', 4.5),
                            ('ink', 'accent-down', 4.5), ('accent', 'field', 4.5), ('accent-down', 'focus', 4.5),
                            ('error', 'base', 4.5), ('border', 'surface', 3), ('text', 'focus', 4.5)]:
        lo, hi = sorted([lum(tokens[fg]), lum(tokens[bg])])
        ratio = (hi+.05)/(lo+.05)
        assert ratio >= minimum, (fg, bg, ratio)
        result[f'{fg}/{bg}'] = round(ratio, 2)
    return result


class Harness:
    def __init__(self):
        self.app = QApplication.instance() or QApplication([])
        self.window = StudioWindow(); self.window.show()
        loop = QEventLoop(); self.window.view.loadFinished.connect(lambda ok: loop.quit()); loop.exec()
        self.wait(800)
        self.target = self.window.view.focusProxy()

    def wait(self, ms):
        end = time.monotonic()+ms/1000
        while time.monotonic() < end:
            self.app.processEvents(); time.sleep(.005)

    def js(self, script):
        box, loop = [], QEventLoop()
        self.window.view.page().runJavaScript(script, 0, lambda r: (box.append(r), loop.quit()))
        QTimer.singleShot(5000, loop.quit); loop.exec()
        return box[0] if box else None

    def jsj(self, script):
        # 数组/对象结果经 JSON 传回（runJavaScript 不转换 JS 数组）
        return json.loads(self.js(f'JSON.stringify({script})'))

    def until(self, script, timeout=180):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            if self.js(script): return True
            self.wait(200)
        raise TimeoutError(script)

    def center(self, selector):
        r = self.jsj(f"(() => {{ const r = document.querySelector({json.dumps(selector)}).getBoundingClientRect(); return [r.left + r.width/2, r.top + r.height/2]; }})()")
        return QPoint(round(r[0]), round(r[1]))

    def reveal(self, selector, block='nearest'):
        self.js(f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block:{json.dumps(block)}}}); 1")
        self.wait(100)

    def click(self, selector, button=Qt.LeftButton):
        self.reveal(selector)
        QTest.mouseClick(self.target, button, Qt.NoModifier, self.center(selector)); self.wait(120)

    def move(self, point, buttons=Qt.NoButton):
        local, global_pos = QPointF(point), QPointF(self.target.mapToGlobal(point))
        self.app.sendEvent(self.target, QMouseEvent(QEvent.MouseMove, local, global_pos, Qt.NoButton, buttons, Qt.NoModifier))
        self.wait(70)

    def dclick(self, point):
        # WebEngine 需要真实的 Press/Release/DblClick/Release 序列才识别为双击
        p, g = QPointF(point), QPointF(self.target.mapToGlobal(point))
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
            buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
            self.app.sendEvent(self.target, QMouseEvent(kind, p, g, Qt.LeftButton, buttons, Qt.NoModifier)); self.wait(20)
        self.wait(120)

    def key(self, key, modifier=Qt.NoModifier):
        QTest.keyClick(self.target, key, modifier); self.wait(80)

    def shot(self, name):
        self.wait(300); self.window.grab().save(str(OUT/(name+'.png')))


def check_gui():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {'contrast': contrast_checks()}
    h = Harness(); w = h.window
    h.js("localStorage.clear(); 1"); w.view.reload(); h.wait(1500); h.target = w.view.focusProxy()
    js = h.js
    assert js("document.querySelector('#dry-has-backing').disabled")

    # 双轴：键盘微调、悬停读数、斜拖同时调整、双击复位。
    field = '.xy-field[data-x=correction]'
    point = field + ' .xy-point'
    h.reveal(field, 'center')
    h.move(QPoint(4, 4))
    assert js("getComputedStyle(document.querySelector('.xy-tip')).visibility === 'hidden'")
    h.move(h.center(point))
    assert js("getComputedStyle(document.querySelector('.xy-tip')).visibility === 'visible'")
    h.js("document.querySelector('#sound-correction').focus(); 1")
    h.key(Qt.Key_Left); assert js('P.correction') == 49
    h.dclick(h.center(point)); assert js('P.correction') == 50
    start = h.center(point)
    distance = round(js("document.querySelector('.xy-scale').getBoundingClientRect().width") * .15)
    QTest.mousePress(h.target, Qt.LeftButton, Qt.NoModifier, start)
    for n in range(1, 5):
        shift = round(distance * n / 4)
        h.move(start + QPoint(shift, -shift), Qt.LeftButton)
    QTest.mouseRelease(h.target, Qt.LeftButton, Qt.NoModifier, start + QPoint(distance, -distance)); h.wait(100)
    assert all(64 <= value <= 66 for value in h.jsj('[P.correction, P.beauty]'))
    h.dclick(h.center(point))
    assert js("localStorage.getItem('sv_param_correction')") == '50'
    assert js('P.beauty') == 50
    h.reveal('#sound-mixdown')
    h.js("document.querySelector('#sound-mixdown').focus(); 1"); h.key(Qt.Key_Left)
    assert js('P.mixdown') == 99
    h.dclick(h.center('#sound-mixdown')); assert js('P.mixdown') == 100

    # 旁路只锁两块面板的纵轴及混音推子。
    h.click('[data-switch=mix_effects]')
    assert js("document.querySelector('#sound-fusion').disabled && document.querySelector('#sound-mixdown').disabled")
    h.shot('mix-bypassed')
    h.click('[data-switch=mix_effects]'); assert js('SW.mix_effects && !document.querySelector("#sound-fusion").disabled')
    # 预设：左键空槽保存；改参数后左键应用；右键待确认，再右键删除
    h.click('.pad[data-slot="3"]'); assert js("JSON.parse(localStorage.getItem('sv_presets'))['3'].params.correction") == 50
    js("document.querySelector('#sound-beauty').focus(); 1"); h.key(Qt.Key_Up); assert js('P.beauty') == 51
    h.click('.pad[data-slot="3"]'); assert js('P.beauty') == 50
    js("presets[9] = {params:{strength:62,target_lufs:-21},switches:{denoise:false,mix_effects:true}}; renderPresets(); 1")
    h.click('.pad[data-slot="9"]')
    assert js("params().strength === 62 && params().denoise === false && hasLegacy('correction')")
    js("document.querySelector('#sound-correction').focus(); 1")
    h.key(Qt.Key_Right)
    assert js("!('strength' in params()) && !('target_lufs' in params())")
    h.click('.pad[data-slot="3"]')
    assert js("!('target_lufs' in params()) && P.correction === 50")
    h.click('.pad[data-slot="3"]', Qt.RightButton); assert js("document.querySelector('.pad[data-slot=\"3\"]').dataset.armed") == 'true'
    h.shot('preset-armed')
    h.click('.pad[data-slot="3"]', Qt.RightButton); assert js("!JSON.parse(localStorage.getItem('sv_presets'))['3']")

    # 缺项：点击主按钮时，缺少的输入闪红框且不开始处理
    h.click('#go'); assert js("document.querySelector('#in-song').classList.contains('need') && !run.busy")

    # 布局：各宽度无横向溢出；两个输入框等宽同行
    layout = {}
    for width, height in ((720, 600), (840, 700), (980, 740), (1080, 780)):
        w.resize(width, height); h.wait(400)
        h.js("document.querySelector('#content').scrollTop = 0; 1")
        m = h.jsj("""(() => { const c = document.querySelector('#content'), a = document.querySelector('#in-song').getBoundingClientRect(),
            b = document.querySelector('.dry-input').getBoundingClientRect();
            return { overflow: c.scrollWidth - c.clientWidth, chipDw: Math.abs(a.width - b.width), chipDy: Math.abs(a.top - b.top),
            switchFits: document.querySelector('#live-preview').getBoundingClientRect().right <
              document.querySelector('#p-meter').getBoundingClientRect().left,
            racks: [...document.querySelectorAll('.rack')].map(r => Math.round(r.getBoundingClientRect().width)) }; })()""")
        assert m['overflow'] <= 0 and m['chipDw'] <= 1 and m['chipDy'] <= 1 and m['switchFits'], (width, m)
        layout[width] = m; h.shot(f'empty-{width}')
    evidence['layout'] = layout
    w.resize(980, 740); h.wait(300)

    # 整曲处理：自动保存 V1，报告与播放器就绪
    folder = Path(tempfile.mkdtemp(prefix='shadowvox-gui-', dir=OUT))
    selftest = ROOT/'tools/_selftest'
    # 拖放：音频落在原曲 / 干音框，文件夹落在输出目录（路径由 Qt 层取得后按落点分配）
    for selector, path in (('#in-song', selftest/'song.wav'), ('#in-dry', selftest/'dry.wav'), ('#out-card', folder)):
        c = h.center(selector)
        w.bridge.filesDropped.emit(json.dumps(dict(paths=[str(path)], x=c.x(), y=c.y()), ensure_ascii=False))
    h.until("inputs.song && inputs.dry && localStorage.getItem('sv_output')", 10)
    assert js("document.querySelector('#in-dry .mq').textContent") == 'dry.wav'
    assert Path(json.loads(js("localStorage.getItem('sv_output')"))) == folder
    assert js("!document.querySelector('#dry-has-backing').disabled && !params().dry_has_backing")
    js("document.querySelector('#dry-has-backing').focus(); 1"); h.key(Qt.Key_Space)
    assert js('params().dry_has_backing')
    h.key(Qt.Key_Space); assert js('!params().dry_has_backing')
    started = time.monotonic(); h.click('#go')
    h.until('run.busy', 10); h.wait(2500); h.shot('processing')
    assert js("document.querySelector('#racks').inert && document.querySelector('#go').classList.contains('running')")
    assert js("document.querySelector('.dry-option').inert")
    h.until('state.ready && !run.busy && !run.settling')
    evidence['first_run_seconds'] = round(time.monotonic()-started, 1)
    assert (folder/'dry_V1.wav').is_file() and (folder/'dry_V1.report.html').is_file()
    report = json.loads((folder/'dry_V1.report.json').read_text(encoding='utf-8'))
    assert report['mixing']['density']['applied']
    assert report['mixing']['engine'] == 'C+' and report['mixing']['readback_verified']
    assert report['mixing']['measured_output_true_peak_dbtp'] <= -.5
    assert report['output'] == str(folder/'dry_V1.wav') and 'vocal_stem' not in report and report['audio_analysis']['output']['lufs'] is not None
    assert js("document.querySelectorAll('#rp-summary dt').length") >= 7
    h.shot('done')

    # 空格播放 / 暂停（真实按键）；切换音源保持位置
    h.js("document.activeElement.blur(); 1"); h.key(Qt.Key_Space); h.wait(700)
    assert w.bridge.player.playing and w.bridge.player.position > 0
    position = w.bridge.player.position
    h.click('.src[data-src="0"]'); h.wait(150); assert w.bridge.player.selected == 0 and w.bridge.player.position >= position
    h.key(Qt.Key_Space); h.wait(150); assert not w.bridge.player.playing

    # 草稿：调整参数后自动切到草稿并在后台更新当前位置
    draft_events = []
    w.bridge.draftState.connect(lambda payload: draft_events.append(json.loads(payload)))
    prior_fusion = js('P.fusion')
    js("document.querySelector('#sound-fusion').focus(); 1"); h.key(Qt.Key_Up)
    assert js('P.fusion') == prior_fusion + 1
    h.until("['ready','error'].includes(document.querySelector('#player').dataset.cache)", 60)
    assert draft_events and draft_events[-1]['state'] == 'ready', draft_events
    assert w.bridge.player.selected == 1 and w.bridge.draft_range[1] > w.bridge.draft_range[0]
    h.shot('draft-ready')

    # 关闭预览后，编辑和跳转只更新参数及播放位置。再次启用才重新生成草稿。
    h.js("document.querySelector('#live-preview').focus(); 1"); h.key(Qt.Key_Space)
    h.until('state.sel === 2 && !switchOn(document.querySelector("#live-preview"))', 10)
    assert not w.bridge.live_preview and js("document.querySelector('.src[data-src=\"1\"]').disabled")
    assert js("localStorage.getItem('sv_live_preview')") == 'false'
    revision = w.bridge.draft_revision
    js("document.querySelector('#sound-fusion').focus(); 1"); h.key(Qt.Key_Up)
    h.click('#p-start'); h.click('.src[data-src="0"]'); h.wait(400)
    assert w.bridge.draft_revision == revision and not w.bridge.draft_timer.isActive()
    assert w.bridge.draft_worker is None and draft_events[-1]['state'] == 'empty'
    h.shot('preview-off-before')
    h.click('.src[data-src="2"]'); h.until('state.sel === 2', 10); h.shot('preview-off-after')
    h.click('#live-preview'); h.until("document.querySelector('#player').dataset.cache === 'ready'", 60)
    assert w.bridge.live_preview and w.bridge.player.selected == 1

    # 第二个版本：后期旁路仍走固定母带，版本菜单可切回 V1。
    h.click('[data-switch=mix_effects]')
    h.click('#go'); h.until('run.busy', 10); h.until('state.ready && !run.busy && !run.settling && current === 2')
    assert (folder/'dry_V2.wav').is_file()
    bypassed = json.loads((folder/'dry_V2.report.json').read_text(encoding='utf-8'))['mixing']
    assert not bypassed['enabled'] and bypassed['engine'] == 'C+' and bypassed['readback_verified']
    assert not bypassed['density']['applied'] and bypassed['measured_output_true_peak_dbtp'] <= -.5
    h.click('#p-ver-btn'); h.wait(300); h.shot('versions')
    h.click('#ver-menu .menu-item:last-child'); h.until('current === 1', 30)
    assert js("document.querySelector('#rp-title').textContent").startswith('V1')
    h.click('#tab-report'); h.wait(900); h.shot('report')
    h.key(Qt.Key_Escape); assert js("document.querySelector('#tab-process').getAttribute('aria-pressed')") == 'true'

    # 弹窗：设置可用 Esc 关闭并把焦点还给品牌按钮
    h.click('#btn-settings'); h.wait(400); assert not js("document.querySelector('#settings').hidden")
    h.shot('settings'); h.key(Qt.Key_Escape); h.wait(300)
    assert js("document.querySelector('#settings').hidden && document.activeElement.id === 'btn-settings'")
    h.click('#dry-has-backing')
    h.until('!state.ready && params().dry_has_backing', 10)
    assert w.bridge.preview_cache is None
    changed_input = folder/'alternate-dry.wav'
    changed_input.write_bytes((selftest/'dry.wav').read_bytes())
    c = h.center('#in-dry')
    w.bridge.filesDropped.emit(json.dumps(dict(paths=[str(changed_input)], x=c.x(), y=c.y())))
    h.until("inputs.dry.endsWith('alternate-dry.wav') && !params().dry_has_backing", 10)

    evidence['passed'] = True
    (OUT/'result.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))
    w.close(); h.app.quit()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    C.DATA_ROOT=OUT/'profile';C.DATA_ROOT.mkdir(exist_ok=True)
    with patch('shadowvox.gui.QSettings',side_effect=lambda *args: QSettings(str(OUT/'window.ini'),QSettings.Format.IniFormat)):
        check_gui()


if __name__ == '__main__':
    run()
