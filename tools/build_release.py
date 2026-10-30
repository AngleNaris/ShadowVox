"""全量构建单文件程序，并记录版本、Git 提交、输入文件及 EXE 校验值。"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', '-c', 'core.excludesFile=',
         '-c', 'core.quotepath=false', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    names = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
    return {name: sha256(ROOT/name) for name in names if (ROOT/name).is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='store_true', help='在 out 中构建候选包，允许尚未提交的源码')
    parser.add_argument('--clean', action='store_true', help='清除构建缓存后重新收集全部依赖')
    args = parser.parse_args()
    version = runpy.run_path(str(ROOT/'src/shadowvox/__init__.py'))['__version__']
    dirty = bool(git('status', '--porcelain'))
    if dirty and not args.candidate:
        raise SystemExit('正式发布必须先提交当前源码；调试构建请使用 --candidate。')
    folder = ROOT/'out/releases'/version
    destination = folder/'candidate' if args.candidate else ROOT/'dist'
    destination.mkdir(parents=True, exist_ok=True)
    executable = destination/f'ShadowVox-{version}.exe'
    if executable.exists() and not args.candidate:
        raise SystemExit(f'{executable.name} 已存在；已发布的版本不能覆盖，请递增版本号。')
    sources = source_hashes()
    external = [ROOT/'models'/name for name in (
        'fcpe.onnx', 'Kim_Vocal_2.onnx', 'whisper-base.pt',
        'hifigan_pkg/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx')]
    external += [ROOT/'tools/vendor/deepfilternet/shadowvox-denoise.exe']
    rubberband = ROOT/'tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows'
    external += [rubberband/'rubberband.exe', rubberband/'sndfile.dll']
    ffmpeg = Path(shutil.which('ffmpeg'))
    external.append(ffmpeg)
    inputs = {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name: sha256(p)
              for p in external}
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm']
    if args.clean:
        command.append('--clean')
    subprocess.run(command+['--distpath', str(destination), '--workpath', str(folder/'pyinstaller'),
                           str(ROOT/'ShadowVox.spec')], cwd=ROOT, check=True)
    if sources != source_hashes():
        raise SystemExit('构建期间源码发生变化，请重新构建。')
    digest = sha256(executable)
    packages = ('pyinstaller', 'numpy', 'scipy', 'soundfile', 'onnxruntime-directml',
                'PySide6-Essentials', 'PySide6-Addons', 'PySide6-WebEngine', 'torch',
                'openai-whisper', 'tiktoken', 'pypinyin', 'pedalboard', 'praat-parselmouth')
    manifest = dict(version=version, release_tag=f'v{version}', git_commit=git('rev-parse', 'HEAD'),
        git_branch=git('branch', '--show-current'), source_dirty=dirty,
        built_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        packages={name: metadata.version(name) for name in packages},
        executable=dict(name=executable.name, bytes=executable.stat().st_size, sha256=digest),
        source_sha256=sources, external_input_sha256=inputs)
    executable.with_suffix('.release.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    executable.with_suffix('.sha256').write_text(f'{digest}  {executable.name}\n', encoding='utf-8')
    print(json.dumps(dict(executable=str(executable), version=version, git_commit=manifest['git_commit'],
                          source_dirty=dirty, sha256=digest), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
