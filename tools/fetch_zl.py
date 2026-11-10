"""准备 ShadowVox 使用的固定版本 ZL Windows VST3。"""
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT/'plugins'
LICENSE = PLUGINS/'ZL-AGPL-3.0.txt'
RELEASES = (
    ('Compressor', '0.6.1', '4e6e14da03cbd4513d890fac01889982e3cd96b2cdeb73678507a09b0c463f7f',
     'da1d8aede0055e031213d97f8e6db4aa472e77a35a050e1e4c1e27c5b013ff4d'),
    ('Limiter', '0.0.1', 'c58ecd0afe2ab2d1040a1fc7c39bd667eb45acdc84ac752602fd2f7af0be1550',
     'd7146a1e9d034418407fcf30d20131fb1694201f3cc41a5ce69e9d08b63c081c'),
)
LICENSE_URL = 'https://raw.githubusercontent.com/spdx/license-list-data/main/text/AGPL-3.0-only.txt'
LICENSE_SHA256 = 'd8a6cc31abc16b6748c7a21f21611f5a1ec33f67d22ca23d7da1c19b95496bee'


def verify(path, digest):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'ZL 输入校验失败：{path}')


def fetch():
    PLUGINS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='shadowvox-zl-') as tmp:
        folder = Path(tmp)
        for name, version, msi_sha, binary_sha in RELEASES:
            bundle_name = f'ZL {name}.vst3'
            binary = Path(bundle_name)/'Contents/x86_64-win'/bundle_name
            if not (PLUGINS/binary).is_file():
                msi = folder/f'ZL.{name}-{version}-Windows-x86-64.msi'
                url = f'https://github.com/ZL-Audio/ZL{name}/releases/download/{version}/{msi.name}'
                urllib.request.urlretrieve(url, msi)
                verify(msi, msi_sha)
                extracted = folder/name
                subprocess.run(['msiexec','/a',str(msi),'/qn',f'TARGETDIR={extracted}'], check=True)
                bundle = next(p for p in extracted.rglob(bundle_name) if p.is_dir())
                verify(bundle.parent/binary, binary_sha)
                shutil.copytree(bundle, PLUGINS/bundle_name, dirs_exist_ok=True)
            verify(PLUGINS/binary, binary_sha)
            print(PLUGINS/binary)
        if not LICENSE.is_file():
            local = folder/LICENSE.name
            urllib.request.urlretrieve(LICENSE_URL,local)
            verify(local,LICENSE_SHA256)
            shutil.copyfile(local,LICENSE)
        verify(LICENSE,LICENSE_SHA256)


if __name__ == '__main__':
    fetch()
