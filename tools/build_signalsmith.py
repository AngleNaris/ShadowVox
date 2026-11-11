"""从仓库内固定版本的源码构建 Windows x64 Signalsmith 渲染助手。"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tools/signalsmith'
OUTPUT = ROOT/'tools/vendor/signalsmith/shadowvox-signalsmith.exe'


def _compiler_setup():
    if shutil.which('cl'):
        return ''
    installer = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    if installer.is_file():
        directory = subprocess.check_output(
            [str(installer), '-latest', '-products', '*', '-requires',
             'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath', '-utf8'],
            encoding='utf-8').strip()
        setup = Path(directory)/'VC/Auxiliary/Build/vcvars64.bat'
        if directory and setup.is_file():
            return f'call "{setup}" >nul || exit /b 1\n'
    raise RuntimeError('请安装带 C++ 桌面开发组件的 Visual Studio Build Tools')


def build():
    setup = _compiler_setup()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUTPUT.parent, prefix='build-') as directory:
        folder = Path(directory)
        compiler = (f'cl /nologo /std:c++17 /O2 /MT /EHsc /DNDEBUG '
                    f'/I"{SOURCE}/third_party/stretch" /I"{SOURCE}/third_party/linear/include" '
                    f'"{SOURCE}/stretch.cpp" /Fe:{OUTPUT.name}')
        (folder/'build.cmd').write_text('@echo off\n'+setup+compiler+'\n', encoding='mbcs')
        subprocess.run(['cmd.exe', '/d', '/c', 'build.cmd'], cwd=folder, check=True)
        (folder/OUTPUT.name).replace(OUTPUT)
    return OUTPUT


if __name__ == '__main__':
    print(build())
