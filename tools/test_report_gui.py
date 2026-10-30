"""真实 QtWebEngine 报告、试听定位、版本切换、保存/取消和离线 PNG 验收。"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))
from PySide6.QtCore import QSettings, QTimer, QUrl, Qt
from PySide6.QtWidgets import QFileDialog, QApplication, QLineEdit
from PySide6.QtTest import QTest
from PySide6.QtGui import QImage
from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
from shadowvox import __version__, config as C
from shadowvox.preview import prepare
from shadowvox.reporting import html_report, visual_payload
from tools import test_gui as base

OUT = Path(os.environ.get('SHADOWVOX_REPORT_TEST_OUT', ROOT/'out/releases'/__version__/'report_gui'))
ACTUAL = ROOT/'out/report_implementation_20261029'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    C.DATA_ROOT = OUT/'profile'; C.DATA_ROOT.mkdir(exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat,QSettings.Scope.UserScope,str(C.DATA_ROOT))
    base.OUT = OUT
    saved = json.loads((ACTUAL/'青花瓷_报告验证.report.json').read_text(encoding='utf-8'))
    payload = visual_payload(saved,saved['audio_analysis'],1,saved['output'])
    tracks,sr,_ = prepare(saved['comparison_before'],saved['output'])
    h = base.Harness(); w = h.window
    h.until('bridge !== null',20)
    second = OUT/'same_settings_V2.wav'
    if not second.exists():
        os.link(saved['output'],second)
    payload2 = copy.deepcopy(payload)
    payload2.update(version=2,path=str(second),name=second.name)
    w.bridge.versions = {1:dict(before=saved['comparison_before'],audio=saved['output'],report=payload),
                        2:dict(before=saved['comparison_before'],audio=str(second),report=payload2)}
    w.bridge.load_tracks(1,tracks,sr,select=2)
    h.js('versions = '+json.dumps([dict(n=i,name=p['name'],path=p['path']) for i,p in ((1,payload),(2,payload2))],ensure_ascii=False)+'; current=1; state.ready=true; renderVersions(); 1')
    h.js('renderReport('+json.dumps(payload,ensure_ascii=False)+', "reveal"); switchTab("report"); 1')
    h.wait(700)
    checks = {'windows':{},'real_audio':True,
              'renderer_sha256':hashlib.sha256((ROOT/'src/shadowvox/ui/report.js').read_bytes()).hexdigest()}
    for width in (720,779,780,781,840,980,1080):
        w.resize(width,880); h.wait(400)
        h.js('document.querySelector(".content").scrollTop=0; 1')
        stats = h.jsj('''(() => { const svg=document.querySelector('#rp-poster svg'), box=svg.getBoundingClientRect();
          const texts=[...svg.querySelectorAll('text')];
          return {width:innerWidth,scroll:document.querySelector('.content').scrollWidth,
            client:document.querySelector('.content').clientWidth, svgWidth:box.width,
            minText:Math.min(...texts.map(t=>Number(t.getAttribute('font-size'))*t.getScreenCTM().a)),
            outOfBounds:texts.filter(t=>{const b=t.getBBox();return b.x < -1 || b.x+b.width>1081;}).map(t=>t.textContent)};
        })()''')
        assert stats['scroll'] <= stats['client']+1,stats
        assert stats['minText'] >= 8,stats
        assert not stats['outOfBounds'],stats
        checks['windows'][str(width)] = stats
        h.shot(f'w{width}-report-top')
        h.js('document.querySelector(".content").scrollTop=700; 1'); h.wait(150)
        h.shot(f'w{width}-report-bottom')
    w.resize(1080,880); h.wait(250)
    w.view.setZoomFactor(1.25); h.wait(300)
    zoom = h.jsj('''(() => {const box=document.querySelector('#rp-poster');return {localScroll:box.scrollWidth,
      localClient:box.clientWidth,pageScroll:document.querySelector('.content').scrollWidth,
      pageClient:document.querySelector('.content').clientWidth};})()''')
    assert zoom['pageScroll'] <= zoom['pageClient']+1,zoom
    checks['zoom_125'] = zoom
    h.shot('zoom125-report')
    w.view.setZoomFactor(1.); h.wait(250)
    h.js('document.querySelector(".content").scrollTop=0; 1')
    # 实际播放器定位；随后停止，避免测试期间持续播放整首。
    h.click('#rp-listen'); h.wait(400)
    assert w.bridge.player.playing
    assert abs(w.bridge.player.position/sr-payload['visual']['pitch']['window']['start']) < 1
    w.bridge.player.pause(); checks['listen'] = True
    h.js('document.querySelector(".content").scrollTop=0; 1'); h.wait(200)
    # 导出异常必须如实反馈；恢复后允许重试，不显示伪造的保存成功。
    h.js('window.__originalReportPng=ShadowReport.png; ShadowReport.png=async()=>{throw new Error("test export error")}; 1')
    h.click('#rp-export'); h.until('!reportExporting',10)
    assert not h.js('document.querySelector("#err-modal").hidden')
    assert '失败' in h.js('document.querySelector("#rp-feedback").textContent')
    h.key(Qt.Key_Escape); h.wait(200)
    h.js('ShadowReport.png=window.__originalReportPng; 1')
    checks['export_error_recovery'] = True
    # 原生槽写入实际 PNG，测试使用可自动操作的 Qt 文件对话框。
    path = OUT/'share.png'; path.unlink(missing_ok=True)
    original_picker = QFileDialog.getSaveFileName
    dialog_state = {'seen':0,'mode':'save'}
    def picker(*args,**kwargs):
        kwargs['options'] = QFileDialog.Option.DontUseNativeDialog
        return original_picker(*args,**kwargs)
    def resolve_dialog():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog,QFileDialog) and not dialog.property('reportTestResolved'):
            dialog.setProperty('reportTestResolved',True)
            dialog_state['seen'] += 1
            if dialog_state['mode'] == 'save':
                dialog.setDirectory(str(OUT))
                def accept():
                    field = dialog.findChild(QLineEdit,'fileNameEdit')
                    field.setFocus(); field.selectAll(); QTest.keyClicks(field,path.name)
                    dialog.accept()
                QTimer.singleShot(250,accept)
            else:
                dialog.reject()
    timer = QTimer(); timer.timeout.connect(resolve_dialog); timer.start(100)
    with patch.object(QFileDialog,'getSaveFileName',side_effect=picker):
        h.click('#rp-export'); h.until("!reportExporting",40)
        assert path.exists() and QImage(str(path)).size().width() == 1080
        assert QImage(str(path)).size().height() == 1440
        assert '已保存' in h.js("document.querySelector('#rp-feedback').textContent")
        old_stat = path.stat().st_mtime_ns
        dialog_state['mode'] = 'cancel'
        h.click('#rp-export'); h.until('!reportExporting',40)
        assert '已取消' in h.js("document.querySelector('#rp-feedback').textContent")
        assert path.stat().st_mtime_ns == old_stat
    timer.stop(); assert dialog_state['seen'] >= 2
    checks['png_save_and_cancel'] = True
    # 版本切换真实载入音频，并由桥接信号替换报告。
    h.click('#p-ver-btn'); h.wait(250)
    h.click('#ver-menu .menu-item:first-child'); h.until('current === 2',40)
    assert h.js('shownReport.version') == 2
    assert 'V2' in h.js("document.querySelector('#rp-title').textContent")
    checks['version_switch'] = True
    h.js('document.querySelector(".content").scrollTop=0; 1')
    h.js('document.querySelector("#rp-listen").focus(); 1'); h.key(Qt.Key_Tab); h.wait(120)
    assert h.js('document.activeElement.id') == 'rp-export'
    focus = h.jsj('''(() => {const s=getComputedStyle(document.querySelector('#rp-export'));return {outline:s.outlineStyle,shadow:s.boxShadow};})()''')
    assert focus['outline'] == 'none' and 'inset' in focus['shadow'],focus
    checks['keyboard_focus'] = focus
    # 离线文件在同一真实浏览器中打开，实际下载 PNG。
    offline = OUT/'current-source.report.html'
    offline.write_text(html_report(saved,saved['audio_analysis'],saved['output']),encoding='utf-8')
    w.view.load(QUrl.fromLocalFile(str(offline)))
    h.until('typeof ShadowReport !== "undefined" && document.querySelector("#report-poster svg") !== null',30)
    h.wait(350); h.shot('offline-report')
    download_path = OUT/'offline-share.png'; download_path.unlink(missing_ok=True)
    downloads = []
    def download(request):
        downloads.append(request)
        request.setDownloadDirectory(str(OUT)); request.setDownloadFileName(download_path.name); request.accept()
    w.view.page().profile().downloadRequested.connect(download)
    h.click('#export-png')
    end = base.time.monotonic()+40
    while (not downloads or not downloads[0].isFinished()) and base.time.monotonic()<end:
        h.wait(100)
    assert downloads and downloads[0].state() == QWebEngineDownloadRequest.DownloadState.DownloadCompleted
    image = QImage(str(download_path))
    assert not image.isNull() and image.width() == 1080 and image.height() == 1440
    checks['offline_png'] = True
    checks['passed'] = True
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=True),flush=True)
    w.close(); h.app.quit()


if __name__ == '__main__':
    main()
