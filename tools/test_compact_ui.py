"""实际 Qt 窗口：紧凑默认、边界、处理顺序、开关范围和尺寸迁移。"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import test_gui as base
from shadowvox import __version__, config as C, gui as gui_module
from shadowvox.gui import DEFAULT_WINDOW, WINDOW_LAYOUT
from PySide6.QtCore import QPoint, QSettings, QSize, Qt

OUT = ROOT/'out/releases'/__version__/'ui'


def measure(h):
    return h.jsj('''(() => {
      const rect=s=>{const r=document.querySelector(s).getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom};};
      const c=document.querySelector('.content'), go=rect('#go'), player=rect('.player');
      return {width:innerWidth,height:innerHeight,left:player.left,right:innerWidth-player.right,
        leftMismatch:Math.abs(go.left-player.left),rightMismatch:Math.abs(go.right-player.right),
        tabMismatch:Math.abs(rect('#tab-report').right-player.right),
        scroll:c.scrollHeight,client:c.clientHeight,overflow:c.scrollWidth-c.clientWidth,
        gap:go.top-Math.max(...[...document.querySelectorAll('.rack')].map(e=>e.getBoundingClientRect().bottom)),
        stages:[...document.querySelectorAll('.rack')].map(e=>e.dataset.rack),
        rows:new Set([...document.querySelectorAll('.rack')].map(e=>Math.round(e.getBoundingClientRect().top))).size,
        clipped:[...document.querySelectorAll('.cell-name,.rack-title,.key-labeled')].filter(e=>{
          const a=e.getBoundingClientRect(), b=e.closest('.cell,.rack-head').getBoundingClientRect();
          return a.left<b.left-1 || a.right>b.right+1;
        }).map(e=>e.textContent)};
    })()''')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    C.DATA_ROOT=OUT/'profile'; C.DATA_ROOT.mkdir(exist_ok=True)
    # 显式组织名的 Qt 构造器仍会使用注册表；这里只将窗口设置指向真实临时 INI。
    def window_settings(*args):
        return QSettings(str(OUT/'window.ini'),QSettings.Format.IniFormat)
    with patch.object(gui_module,'QSettings',side_effect=window_settings):
        check(window_settings())


def check(settings):
    settings.remove('window/layout')
    settings.setValue('window/size',QSize(1080,780)); settings.setValue('window/pos',QPoint(40,40))
    settings.sync()
    os.environ.pop('SHADOWVOX_WINDOW',None)
    base.OUT=OUT
    h=base.Harness(); w=h.window
    h.until('bridge !== null',20)
    h.js('localStorage.clear(); 1');w.view.reload();h.wait(1000);h.target=w.view.focusProxy()
    h.until('bridge !== null',20)
    assert (w.width(),w.height())==DEFAULT_WINDOW
    evidence={'default':measure(h),'old_geometry_migrated':True}
    assert evidence['default']['scroll']<=evidence['default']['client']+1
    assert evidence['default']['rows']==2
    assert 12<=evidence['default']['gap']<=40
    assert evidence['default']['stages']==['cleanup','tune','level','tone','mix','output']
    assert h.js('document.querySelector(".cell[data-key=declip] .cell-name").textContent')=='削波修复'
    assert h.js('document.querySelector("[data-rack=output] .control-group-title").textContent')=='人声 + 伴奏'
    h.shot('default')

    h.click('[data-switch=mix_effects]')
    assert h.js('[...document.querySelectorAll(".rack-post .rack-body")].every(e=>e.inert)')
    assert not h.js('document.querySelector(".cell[data-key=vocal_gain_db]").inert')
    old=h.js('P.target_lufs')
    h.js('document.querySelector(".cell[data-key=target_lufs] .knob-wrap").focus(); 1')
    h.key(Qt.Key_Left); assert h.js('P.target_lufs')==old
    h.js('document.querySelector(".cell[data-key=vocal_gain_db] .knob-wrap").focus(); 1')
    gain=h.js('P.vocal_gain_db'); h.key(Qt.Key_Right)
    assert h.js('P.vocal_gain_db')==gain+.5
    h.shot('post-bypassed')
    h.click('[data-switch=mix_effects]')
    assert h.js('[...document.querySelectorAll(".rack-post .rack-body")].every(e=>!e.inert)')
    evidence['post_bypass_scope']=True
    h.click('[data-help=output]'); h.wait(250)
    assert h.js('document.querySelector("#help").textContent.includes("人声和伴奏")')
    h.shot('output-help'); h.key(Qt.Key_Escape)
    assert h.js('document.activeElement.dataset.help')=='output'
    evidence['output_help_and_focus']=True

    evidence['windows']={}
    for width,height in [(720,600),(897,680),(898,680),(899,680),(900,680),(1080,780)]:
        w.resize(width,height); h.wait(350)
        h.js('document.querySelector(".content").scrollTop=0; 1')
        stats=measure(h)
        assert stats['overflow']<=1 and not stats['clipped'],stats
        assert abs(stats['left']-stats['right'])<=1,stats
        assert max(stats['leftMismatch'],stats['rightMismatch'],stats['tabMismatch'])<=1,stats
        if width==899:
            assert stats['rows']==2 and stats['scroll']<=stats['client']+1,stats
        evidence['windows'][f'{width}x{height}']=stats
        h.shot(f'window-{width}')
        if width==720:
            h.js('document.querySelector(".content").scrollTop=9999; 1');h.wait(200)
            h.shot('minimum-bottom')
    w.resize(*DEFAULT_WINDOW);w.view.setZoomFactor(1.25);h.wait(350)
    zoom=measure(h)
    assert zoom['overflow']<=1 and not zoom['clipped'],zoom
    assert max(zoom['leftMismatch'],zoom['rightMismatch'])<=1,zoom
    h.shot('zoom125');evidence['zoom125']=zoom
    w.view.setZoomFactor(1.);w.resize(960,700);h.wait(200);w.close()
    settings.sync()
    assert settings.value('window/layout',0,type=int)==WINDOW_LAYOUT
    C.DATA_ROOT=OUT/'reopen-profile';C.DATA_ROOT.mkdir(exist_ok=True)
    second=base.Harness()
    assert (second.window.width(),second.window.height())==(960,700)
    second.window.close();second.app.quit()
    evidence['manual_geometry_restored']=True;evidence['passed']=True
    (OUT/'checks.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
