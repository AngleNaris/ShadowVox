"""真实窗口截图：启动 WebView 工作台，按脚本执行 JS 并截图，用于界面验收。

用法：python tools/shot_gui.py 输出目录 宽x高 [步骤 JSON 文件]
步骤为 [{"js": "...", "wait": 毫秒, "shot": "名称", "py": "bridge 上的 Python 表达式"}]。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    out, size = Path(sys.argv[1]), sys.argv[2]
    steps = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')) if len(sys.argv) > 3 else [{'wait': 900, 'shot': 'start'}]
    out.mkdir(parents=True, exist_ok=True)
    os.environ['SHADOWVOX_WINDOW'] = size
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from shadowvox.gui import StudioWindow
    app = QApplication(sys.argv[:1])
    window = StudioWindow(); window.show()
    bridge, log = window.bridge, []
    console = out/'console.log'; console.write_text('', encoding='utf-8')
    def js_log(level, message, line, source):
        with console.open('a', encoding='utf-8') as f: f.write(f'{level.name} {Path(source).name}:{line} {message}\n')
    window.view.page().javaScriptConsoleMessage = js_log
    queue = list(steps)

    def step():
        if not queue:
            (out/'console.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
            window.bridge.shutdown(); app.quit(); return
        s = queue.pop(0)
        if 'py' in s:
            exec(s['py'], dict(bridge=bridge, window=window, ROOT=ROOT))
        if 'js' in s:
            window.view.page().runJavaScript(s['js'], 0, lambda r: log.append([s.get('shot') or s['js'][:40], r]))
        def after():
            if 'shot' in s:
                window.grab().save(str(out/(s['shot']+'.png')))
            step()
        def poll(left=s.get('timeout', 300000)):
            # until：轮询 JS 条件直到为真（例如等待处理完成）
            def check(ok):
                if ok or left <= 0: QTimer.singleShot(s.get('wait', 300), after)
                else: QTimer.singleShot(500, lambda: poll(left-500))
            window.view.page().runJavaScript(s['until'], 0, check)
        if 'until' in s: QTimer.singleShot(200, poll)
        else: QTimer.singleShot(s.get('wait', 300), after)
    window.view.loadFinished.connect(lambda ok: QTimer.singleShot(400, step))
    app.exec()


if __name__ == '__main__':
    main()
