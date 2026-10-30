"""用 Python 3.12 构建独立 AI 降噪进程；输出无需目标机器安装 Python。"""
from pathlib import Path
from importlib import resources

root = Path(SPECPATH).parent
model = Path(resources.files("deepfilternet_rs.models"))
a = Analysis([str(root/"tools/denoise_helper.py")],
             datas=[(str(model),"deepfilternet_rs/models")],
             hiddenimports=["deepfilternet_rs.models"],
             excludes=["torch","scipy","matplotlib","pandas"])
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,a.binaries,a.datas,[],name="shadowvox-denoise",console=True,
          debug=False,strip=False,upx=False)
