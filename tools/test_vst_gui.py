"""设置中的 VST3 管理（桥接层）和本机 VST3 窗口/状态保存验收，不处理音频。"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtTest import QTest
from shadowvox import config
from shadowvox.gui_bridge import Bridge


def plugin_windows(pid):
    if not pid:
        return []
    found = []
    user32 = ctypes.windll.user32
    kernel = ctypes.windll.kernel32
    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size",wintypes.DWORD),("usage",wintypes.DWORD),("pid",wintypes.DWORD),
                    ("heap",ctypes.c_size_t),("module",wintypes.DWORD),("threads",wintypes.DWORD),
                    ("parent",wintypes.DWORD),("priority",ctypes.c_long),("flags",wintypes.DWORD),
                    ("name",wintypes.WCHAR*260)]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(2,0)
    entry = ProcessEntry(); entry.size = ctypes.sizeof(entry)
    parents = {}
    more = kernel.Process32FirstW(snapshot,ctypes.byref(entry))
    while more:
        parents[entry.pid] = entry.parent
        more = kernel.Process32NextW(snapshot,ctypes.byref(entry))
    kernel.CloseHandle(snapshot)
    descendants = {pid}
    while True:
        extended = descendants | {child for child,parent in parents.items() if parent in descendants}
        if extended == descendants:
            break
        descendants = extended
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
    @callback_type
    def callback(hwnd, _):
        owner = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
        if owner.value in descendants and user32.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd,title,len(title))
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd,cls,len(cls))
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd,ctypes.byref(rect))
            width, height = rect.right-rect.left, rect.bottom-rect.top
            if width > 100 and height > 100 and not cls.value.startswith("UAC"):
                found.append((hwnd,title.value,cls.value,width,height))
        return True
    user32.EnumWindows(callback,0)
    return found


def run():
    app = QApplication([])
    plugin = Path(r"C:\Program Files\Common Files\VST3\iZotope\RX 11 De-clip.vst3")
    with tempfile.TemporaryDirectory() as folder, patch.object(config,"DATA_ROOT",Path(folder)):
        bridge = Bridge(); status = []
        bridge.vstStatus.connect(lambda text: status.append(json.loads(text)))
        assert json.loads(bridge.vstList()) == []
        with patch.object(QFileDialog,"getExistingDirectory",return_value=str(plugin)):
            bridge.vstAdd("folder"); listed = json.loads(bridge.vstAdd("folder"))
        assert len(listed) == 1 and listed[0]["name"] == plugin.stem, listed
        with patch.object(QFileDialog,"getOpenFileName",return_value=(folder,"")):
            bridge.vstAdd("file")
        assert "VST3" in status[-1]["text"] and len(json.loads(bridge.vstList())) == 1
        if plugin.exists():
            key = hashlib.sha256(os.path.normcase(str(plugin.resolve())).encode("utf-8")).hexdigest()[:20]
            parameters = Path(folder)/"settings/vst3"/(key+".params.json")
            parameters.write_text(json.dumps({"clipping_threshold":.4}),encoding="utf-8")
            if len(sys.argv)>1:
                executable = str(Path(sys.argv[1]).resolve())
                with patch.object(sys,"frozen",True,create=True), patch.object(sys,"executable",executable):
                    bridge.vstOpen(0)
            else:
                bridge.vstOpen(0)
            process = bridge.plugin_process
            assert status[-1]["busy"]
            deadline = time.monotonic()+30
            handles = []
            while time.monotonic()<deadline and not handles:
                QTest.qWait(100)
                if process.state()==QProcess.NotRunning:
                    raise AssertionError(status[-1]["text"])
                handles = plugin_windows(process.processId())
            assert handles,"插件界面未显示"
            print("插件进程窗口：",[(x[1],x[2],x[3],x[4]) for x in handles],flush=True)
            assert "先关闭" in bridge.can_close()
            deadline = time.monotonic()+30
            seen = set()
            while process.state()!=QProcess.NotRunning and time.monotonic()<deadline:
                for handle in plugin_windows(process.processId()):
                    if handle[0] not in seen:
                        print("关闭测试窗口：",handle[1],flush=True)
                        seen.add(handle[0])
                        ctypes.windll.user32.PostMessageW(handle[0],0x0010,0,0)
                QTest.qWait(100)
            QTest.qWait(100)
            assert process.state()==QProcess.NotRunning,"插件进程未关闭"
            assert process.exitCode()==0,status[-1]["text"]
            assert "已保存" in status[-1]["text"],status[-1]["text"]
            state = next((Path(folder)/"settings/vst3").glob("*.state"))
            assert state.stat().st_size>0
            saved = json.loads(parameters.read_text(encoding="utf-8"))
            assert abs(saved['clipping_threshold']-.4)<.002,saved['clipping_threshold']
            print("VST3 窗口打开并关闭，状态与参数已恢复并保存：",state.stat().st_size,"bytes")
        else:
            print("本机 RX 11 De-clip 不存在，跳过真实插件窗口检查")
        assert bridge.can_close() == ""
        bridge.shutdown()


if __name__=="__main__":
    run()
