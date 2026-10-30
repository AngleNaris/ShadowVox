from pathlib import Path
import shutil
import sys
import os
import runpy
import importlib.metadata as metadata
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct)

root=Path(SPECPATH)
version=runpy.run_path(str(root/'src/shadowvox/__init__.py'))['__version__']
filename=f'ShadowVox-{version}'
version_info=VSVersionInfo(
    ffi=FixedFileInfo(filevers=tuple(map(int,version.split('.')))+(0,),
                      prodvers=tuple(map(int,version.split('.')))+(0,), fileType=1),
    kids=[StringFileInfo([StringTable('080404B0',[
        StringStruct('FileDescription','ShadowVox 修音工作台'),
        StringStruct('ProductName','ShadowVox'),
        StringStruct('FileVersion',version), StringStruct('ProductVersion',version),
        StringStruct('OriginalFilename',filename+'.exe')])]),
        VarFileInfo([VarStruct('Translation',[2052,1200])])])
datas=[(str(root/'THIRD_PARTY.txt'),'licenses'),(str(root/'models/fcpe.onnx'),'models'),
       (str(root/'models/Kim_Vocal_2.onnx'),'models'),
       (str(root/'models/whisper-base.pt'),'models'),
       (str(root/'models/hifigan_pkg/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx'),'models/hifigan_pkg'),
       (str(root/'assets/shadowvox.ico'),'assets'),
       (str(root/'src/shadowvox/ui'),'shadowvox/ui'),
       (str(root/'src/shadowvox/vendor/synctoolbox/LICENSE'),'licenses/synctoolbox'),
       (str(root/'tools/vendor/deepfilternet/licenses'),'licenses/deepfilternet')]
# 整句对齐与 R3 变速：官方 Rubber Band 命令行（GPL-2.0-or-later）及其 libsndfile。
rubberband=root/'tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows'
datas+=[(str(rubberband/'COPYING.txt'),'licenses/rubberband')]
for path in (root/'models/hifigan_pkg').glob('*.txt'):
    datas.append((str(path),'licenses/hifigan'))
datas+=copy_metadata('onnxruntime-directml')
datas+=collect_data_files('whisper')
for package in ('numpy','scipy','PySide6','shiboken6'):
    datas+=collect_data_files(package,includes=['**/LICENSE*','**/COPYING*','**/licenses/*'])
for package in ('numpy','scipy','soundfile','sounddevice','onnxruntime-directml','PySide6-Essentials','PySide6-Addons','PySide6-WebEngine','shiboken6','praat-parselmouth','numba','llvmlite','torch','openai-whisper','tiktoken','pypinyin','pedalboard'):
    dist=metadata.distribution(package)
    for relative in dist.files or []:
        if 'LICENSE' in str(relative).upper() or 'COPYING' in str(relative).upper():
            source=Path(dist.locate_file(relative))
            if source.is_file():datas.append((str(source),'licenses/'+package+'/'+str(relative.parent)))
ffmpeg=Path(shutil.which('ffmpeg')).resolve()
datas.append((str(ffmpeg.parent.parent/'LICENSE'),'licenses/ffmpeg'))
python_license=Path(sys.base_prefix)/'LICENSE.txt'
if python_license.exists():datas.append((str(python_license),'licenses/python'))
binaries=[(str(ffmpeg),'bin'),(str(rubberband/'rubberband.exe'),'bin'),(str(rubberband/'sndfile.dll'),'bin'),
          (str(root/'tools/vendor/deepfilternet/shadowvox-denoise.exe'),'bin')]+collect_dynamic_libs('onnxruntime')
qt_runtime=Path(metadata.distribution('PySide6-Essentials').locate_file('PySide6'))
windows=Path(os.environ['SystemRoot'])
# 构建进程继承的外部工具 PATH 可能提供不兼容的同名 ICU/UCRT DLL。
os.environ['PATH']=os.pathsep.join(map(str,[Path(sys.executable).parent,sys.base_prefix,
                                          qt_runtime,windows/'System32',windows]))
a=Analysis([str(root/'desktop.py')],pathex=[str(root/'src')],datas=datas,binaries=binaries,
           hiddenimports=['tiktoken_ext.openai_public','PySide6.QtWebEngineWidgets','PySide6.QtWebEngineCore','PySide6.QtWebChannel'],
           excludes=['torchvision','torchaudio','audio_separator','onnx2torch','onnx',
                     'librosa','sklearn','matplotlib','pandas','IPython','pytest','PyQt5','PyQt6','PySide2',
                     'synctoolbox','libfmp','music21','pretty_midi'],
           noarchive=False)
# PATH 中其他工具的旧 VC 运行库会先于 Qt 同名 DLL 加载，导致 QtCore 缺入口。
# 所有位置统一使用 PySide6 随包分发的同版本运行库，包括 EXE 解包根目录。
runtime_files={p.name.lower():p for pattern in ('msvcp140*.dll','vcruntime140*.dll')
               for p in qt_runtime.glob(pattern)}
a.binaries=[(dest,str(runtime_files[Path(dest).name.lower()]),kind)
            if Path(dest).name.lower() in runtime_files else (dest,source,kind)
            for dest,source,kind in a.binaries]
# Windows 10 自带这些系统 DLL；不得用 Poppler/libheif 的兼容层覆盖。
a.binaries=[item for item in a.binaries if Path(item[0]).name.lower() not in
            ('icuuc.dll','icuin.dll','icudt.dll','ucrtbase.dll') and not
            Path(item[0]).name.lower().startswith(('api-ms-win-','ext-ms-win-'))]
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name=filename,version=version_info,debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=False,
        disable_windowed_traceback=False,icon=str(root/'assets/shadowvox.ico'))
