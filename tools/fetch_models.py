"""下载运行所需的模型到 models/。

用法: .venv\\Scripts\\python.exe tools\\fetch_models.py
"""
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FCPE_URL = ("https://raw.githubusercontent.com/KCKT0112/HachiTune/"
            "d68808c37dede264d80d2d1050dce27af34ad654/Resources/models/fcpe.onnx")
FCPE_SHA256 = "013b507acd406de122b61ee497300ac6be082511101bc2136b886e48d9737e5d"
HIFIGAN_URL = ("https://github.com/openvpi/vocoders/releases/download/"
               "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
               "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep")
VOCAL_URL = ("https://github.com/TRvlvr/model_repo/releases/download/"
             "all_public_uvr_models/Kim_Vocal_2.onnx")
WHISPER_SHA256 = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
WHISPER_URL = f"https://openaipublic.azureedge.net/main/whisper/models/{WHISPER_SHA256}/base.pt"


def download(url, dest, expected_sha=None):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  已存在 {dest.name} ({dest.stat().st_size / 1e6:.1f}MB)")
        return dest
    print(f"  下载 {dest.name} …")
    with urlopen(url) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {done / total:5.1%}", end="", flush=True)
    print(f"\r   完成 {dest.stat().st_size / 1e6:.1f}MB")
    if expected_sha:
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha != expected_sha:
            dest.unlink()
            raise SystemExit(f"SHA256 不匹配，已删除: {sha}")
    return dest


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    download(FCPE_URL, MODELS / "fcpe.onnx", FCPE_SHA256)
    download(VOCAL_URL, MODELS / "mdx_vocals.onnx")
    download(WHISPER_URL, MODELS / "whisper-base.pt", WHISPER_SHA256)

    oudep = MODELS / "hifigan.oudep"
    download(HIFIGAN_URL, oudep)
    pkg = MODELS / "hifigan_pkg"
    onnx = pkg / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
    if not onnx.exists():
        print("  解包声码器 …")
        pkg.mkdir(exist_ok=True)
        zipfile.ZipFile(oudep).extractall(pkg)
    print("全部就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
