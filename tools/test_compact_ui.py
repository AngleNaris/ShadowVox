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
from PySide6.QtTest import QTest

OUT = ROOT/'out/releases'/__version__/'ui'


def measure(h):
    return h.jsj('''(() => {
      const rect=s=>{const r=document.querySelector(s).getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom};};
      const c=document.querySelector('.content'), go=rect('#go'), player=rect('.player');
      const pads=[...document.querySelectorAll('.xy-field')].map(e=>e.getBoundingClientRect());
      const content=c.getBoundingClientRect(), mix=rect('.mix-row');
      return {width:innerWidth,height:innerHeight,left:player.left,right:innerWidth-player.right,
        leftMismatch:Math.abs(go.left-player.left),rightMismatch:Math.abs(go.right-player.right),
        tabMismatch:Math.abs(rect('#tab-report').right-player.right),
        scroll:c.scrollHeight,client:c.clientHeight,overflow:c.scrollWidth-c.clientWidth,
        gap:go.top-Math.max(...[...document.querySelectorAll('.rack')].map(e=>e.getBoundingClientRect().bottom)),
        stages:[...document.querySelectorAll('.axis-label')].map(e=>e.textContent),
        squares:pads.map(a=>[a.width,a.height]), sameRow:pads[0].top===pads[1].top,
        mixBelow:mix.top>=Math.max(...pads.map(a=>a.bottom)),
        allControlsVisible:pads.every(a=>a.top>=content.top-1) && mix.bottom<=content.bottom+1,
        clipped:[...document.querySelectorAll('.axis-label,.rack-title,.key-labeled span,.chip:not([data-filled="true"]) .mq')].filter(e=>{
          const a=e.getBoundingClientRect(), b=e.closest('.xy-control,.mix-row,.rack-head,.key-labeled,.chip-name').getBoundingClientRect();
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
    assert evidence['default']['stages']==['修正','美化','融合','比例','混音']
    assert h.js('P.correction === 50 && P.beauty === 50 && P.fusion === 50 && P.ratio === 0 && P.mixdown === 100')
    assert h.js('Array.from(document.querySelectorAll(".xy-field")).every(e => e.style.getPropertyValue("--x") === "50%" && e.style.getPropertyValue("--y") === "50%")')
    assert h.js('Array.from(document.querySelectorAll("[data-sound]")).every(input => input.step === "1")')
    h.shot('default')

    h.click('[data-switch=mix_effects]')
    assert h.js('[...document.querySelectorAll("#sound-beauty,#sound-fusion,#sound-mixdown")].every(e=>e.disabled)')
    assert h.js('document.querySelectorAll(".xy-field.is-x-only").length === 2 && !document.querySelector("#sound-ratio").disabled')
    old=h.js('P.mixdown')
    h.js('document.querySelector("#sound-mixdown").focus(); 1')
    h.key(Qt.Key_Left); assert h.js('P.mixdown')==old
    h.js('document.querySelector("#sound-ratio").focus(); 1')
    gain=h.js('P.ratio'); h.key(Qt.Key_Right)
    assert h.js('P.ratio')==gain+1
    ratio_field='.xy-field[data-x=ratio]'
    h.reveal(ratio_field,'center')
    point=h.center(ratio_field)+QPoint(45,-60)
    QTest.mouseClick(h.target,Qt.LeftButton,Qt.NoModifier,point);h.wait(120)
    assert h.js('P.fusion === 50 && P.ratio > 1')
    h.shot('post-bypassed')
    h.click('[data-switch=mix_effects]')
    assert h.js('[...document.querySelectorAll("#sound-beauty,#sound-fusion,#sound-mixdown")].every(e=>!e.disabled)')
    h.reveal(ratio_field,'center');h.dclick(h.center(ratio_field+' .xy-point'))
    assert h.js('P.ratio === 0')
    evidence['post_bypass_scope']=True
    h.click('[data-help=sound]'); h.wait(250)
    assert h.js('document.querySelector("#help").textContent.includes("混响、延时和伴奏避让")')
    h.shot('output-help'); h.key(Qt.Key_Escape)
    assert h.js('document.activeElement.dataset.help')=='sound'
    evidence['output_help_and_focus']=True

    evidence['windows']={}
    for width,height in [(720,600),(900,680),(953,680),(954,680),(955,680),(1080,780)]:
        w.resize(width,height); h.wait(350)
        h.js('document.querySelector(".content").scrollTop=0; 1')
        stats=measure(h)
        assert stats['overflow']<=1 and not stats['clipped'],stats
        assert abs(stats['left']-stats['right'])<=1,stats
        assert max(stats['leftMismatch'],stats['rightMismatch'],stats['tabMismatch'])<=1,stats
        assert stats['sameRow'] and stats['mixBelow'],stats
        assert all(abs(w-h)<=1 for w,h in stats['squares']) and abs(stats['squares'][0][0]-stats['squares'][1][0])<=1,stats
        evidence['windows'][f'{width}x{height}']=stats
        h.shot(f'window-{width}')
        h.js('document.querySelector(".content").scrollTop=9999; 1');h.wait(200)
        at_bottom=measure(h)
        assert at_bottom['allControlsVisible'] and 12<=at_bottom['gap']<=40,at_bottom
        if width in (720,900):
            h.shot(f'controls-{width}')
        if width==900:
            h.move(h.center(ratio_field+' .xy-point'))
            assert h.js('getComputedStyle(document.querySelector(".xy-field[data-x=ratio] .xy-tip")).visibility === "visible"')
            h.shot('controls-hover')
            thumb=h.jsj('(()=>{const r=document.querySelector(".mix-marks").getBoundingClientRect();return [r.right,document.querySelector("#sound-mixdown").getBoundingClientRect().y+13]})()')
            h.move(QPoint(round(thumb[0]),round(thumb[1])))
            assert h.js('getComputedStyle(document.querySelector(".mix-tip")).visibility === "visible"')
            h.move(QPoint(4,4))
    w.resize(*DEFAULT_WINDOW);w.view.setZoomFactor(1.25);h.wait(350)
    zoom=measure(h)
    assert zoom['overflow']<=1 and not zoom['clipped'],zoom
    assert max(zoom['leftMismatch'],zoom['rightMismatch'])<=1,zoom
    h.shot('zoom125');evidence['zoom125']=zoom
    h.js('document.querySelector(".content").scrollTop=9999; 1');h.wait(200)
    assert measure(h)['allControlsVisible']
    h.shot('controls-zoom125')
    dry = ROOT/'tools/_selftest/dry.wav'
    h.js(f"chooseInput('dry', {json.dumps(str(dry))}); 1")
    h.until("inputs.dry && !document.querySelector('#dry-has-backing').disabled",10)
    h.js("document.querySelector('#dry-has-backing').focus(); 1")
    h.key(Qt.Key_Space)
    assert h.js("params().dry_has_backing && document.activeElement.id === 'dry-has-backing'")
    h.shot('dry-backing-checked')
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
