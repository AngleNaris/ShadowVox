"""下载并校验 ShadowVox 运行与打包所需的全部离线模型到 models/。

用法:
  .venv\\Scripts\\python.exe tools\\fetch_models.py
  .venv\\Scripts\\python.exe tools\\fetch_models.py --verify-only
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

FCPE_URL = (
    "https://raw.githubusercontent.com/KCKT0112/HachiTune/"
    "d68808c37dede264d80d2d1050dce27af34ad654/Resources/models/fcpe.onnx"
)
FCPE_SHA256 = "013b507acd406de122b61ee497300ac6be082511101bc2136b886e48d9737e5d"

VOCAL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/Kim_Vocal_2.onnx"
)
VOCAL_SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"

WHISPER_SHA256 = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
WHISPER_URL = (
    f"https://openaipublic.azureedge.net/main/whisper/models/{WHISPER_SHA256}/base.pt"
)

HIFIGAN_URL = (
    "https://github.com/openvpi/vocoders/releases/download/"
    "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep"
)
HIFIGAN_ARCHIVE_SHA256 = "ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e"
HIFIGAN_ONNX_NAME = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
HIFIGAN_ONNX_SHA256 = "b297c0c790dbcfe6b2123d87cc20bf8718969f91ffa3170a198c8fb53686155d"

HUBERTFA_URL = (
    "https://github.com/wolfgitpr/HubertFA/releases/download/"
    "v0.0.7/1218_hfa_model_new_dict.zip"
)
HUBERTFA_ARCHIVE_SHA256 = "48bd6dbcc293e47cc6cbcc556baf575e3d91d61d0ac88ddeccd6098f8f346fa2"
HUBERTFA_ONNX_SHA256 = "4722c13e1d5c1d740ec69c9121f5ec37a88d534aba7637bafb480208be170183"
HUBERTFA_REQUIRED = ("model.onnx", "config.json", "vocab.json", "ds-zh-pinyin-lite.txt")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected_sha=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 1024:
        return False
    if expected_sha and sha256_file(path) != expected_sha:
        return False
    return True


def download(url, dest, expected_sha=None, *, force=False):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and verify_file(dest, expected_sha):
        print(f"  [OK] 已存在且校验通过 {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    if dest.exists() and not force:
        print(f"  [WARN] {dest.name} 不完整或校验不匹配，将重新下载 …")

    print(f"  [DL] 下载 {dest.name} …")
    part = dest.with_name(dest.name + ".part")
    req = Request(url, headers={"User-Agent": "ShadowVox-ModelFetcher"})
    try:
        with urlopen(req) as resp, part.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {done / total:5.1%} ({done / 1e6:.1f}/{total / 1e6:.1f} MB)",
                          end="", flush=True)
        print(f"\r   完成 {part.stat().st_size / 1e6:.1f} MB" + " " * 20)
        if expected_sha:
            actual = sha256_file(part)
            if actual != expected_sha:
                raise SystemExit(
                    f"SHA256 校验失败 ({dest.name}):\n  期望: {expected_sha}\n  实际: {actual}"
                )
        part.replace(dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    return dest


def _safe_extract_flat(archive_path, target_dir, anchor_filename):
    """定位压缩包内含 anchor_filename 的目录，将其下所有文件安全解包到 target_dir。"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        anchor = next((m for m in members if Path(m.filename).name == anchor_filename), None)
        if anchor is None:
            raise SystemExit(f"压缩包中未找到 {anchor_filename}: {archive_path}")
        prefix = anchor.filename[:-len(anchor_filename)]
        resolved_root = target_dir.resolve()
        for member in members:
            if not member.filename.startswith(prefix):
                continue
            rel = member.filename[len(prefix):]
            if not rel:
                continue
            dest = (target_dir / rel).resolve()
            if not dest.is_relative_to(resolved_root):
                raise SystemExit(f"拒绝不安全的压缩包路径: {member.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def verify_all():
    """返回 [(label, ok, detail), ...]。"""
    checks = []
    for label, path, sha in (
        ("FCPE 音高模型 (models/fcpe.onnx)", MODELS / "fcpe.onnx", FCPE_SHA256),
        ("Kim Vocal 2 人声分离 (models/Kim_Vocal_2.onnx)", MODELS / "Kim_Vocal_2.onnx", VOCAL_SHA256),
        ("Whisper Base 识别 (models/whisper-base.pt)", MODELS / "whisper-base.pt", WHISPER_SHA256),
        ("PC-NSF-HiFiGAN 声码器 (models/hifigan_pkg)", MODELS / "hifigan_pkg" / HIFIGAN_ONNX_NAME, HIFIGAN_ONNX_SHA256),
        ("HubertFA 中文对齐模型 (models/hubertfa/model.onnx)", MODELS / "hubertfa" / "model.onnx", HUBERTFA_ONNX_SHA256),
    ):
        if not path.is_file():
            checks.append((label, False, "缺失"))
        elif sha256_file(path) != sha:
            checks.append((label, False, "SHA256 不匹配"))
        else:
            checks.append((label, True, f"{path.stat().st_size / 1e6:.1f} MB"))

    missing_hfa = [name for name in HUBERTFA_REQUIRED if not (MODELS / "hubertfa" / name).is_file()]
    if missing_hfa:
        checks.append(("HubertFA 词典与配置 (models/hubertfa/)", False, f"缺失: {', '.join(missing_hfa)}"))
    else:
        checks.append(("HubertFA 词典与配置 (models/hubertfa/)", True, "完整"))
    return checks


def fetch_all(*, force=False):
    MODELS.mkdir(parents=True, exist_ok=True)

    # 1. FCPE
    download(FCPE_URL, MODELS / "fcpe.onnx", FCPE_SHA256, force=force)

    # 2. Kim Vocal 2（兼容旧版脚本下载的 mdx_vocals.onnx，避免重复下载）
    vocal_dest = MODELS / "Kim_Vocal_2.onnx"
    legacy_vocal = MODELS / "mdx_vocals.onnx"
    if not force and not verify_file(vocal_dest, VOCAL_SHA256) and verify_file(legacy_vocal, VOCAL_SHA256):
        print("  [COPY] 从已有 models/mdx_vocals.onnx 复用校验通过的 Kim_Vocal_2.onnx")
        shutil.copy2(legacy_vocal, vocal_dest)
    download(VOCAL_URL, vocal_dest, VOCAL_SHA256, force=force)

    # 3. Whisper Base
    download(WHISPER_URL, MODELS / "whisper-base.pt", WHISPER_SHA256, force=force)

    # 4. PC-NSF-HiFiGAN
    hifigan_pkg = MODELS / "hifigan_pkg"
    hifigan_onnx = hifigan_pkg / HIFIGAN_ONNX_NAME
    hifigan_notices = list(hifigan_pkg.glob("*.txt")) if hifigan_pkg.is_dir() else []
    if force or not verify_file(hifigan_onnx, HIFIGAN_ONNX_SHA256) or not hifigan_notices:
        oudep = MODELS / "hifigan.oudep"
        download(HIFIGAN_URL, oudep, HIFIGAN_ARCHIVE_SHA256, force=force)
        print("  [UNZIP] 解包 PC-NSF-HiFiGAN 声码器 …")
        _safe_extract_flat(oudep, hifigan_pkg, HIFIGAN_ONNX_NAME)
        if not verify_file(hifigan_onnx, HIFIGAN_ONNX_SHA256):
            raise SystemExit(f"解包后的声码器校验失败: {hifigan_onnx}")
    else:
        print(f"  [OK] 已存在且校验通过 hifigan_pkg/{HIFIGAN_ONNX_NAME} ({hifigan_onnx.stat().st_size / 1e6:.1f} MB)")

    # 5. HubertFA 中文歌声对齐模型
    hfa_dir = MODELS / "hubertfa"
    hfa_ready = (
        not force
        and verify_file(hfa_dir / "model.onnx", HUBERTFA_ONNX_SHA256)
        and all((hfa_dir / name).is_file() for name in HUBERTFA_REQUIRED)
    )
    if hfa_ready:
        print(f"  [OK] 已存在且校验通过 hubertfa/model.onnx ({(hfa_dir / 'model.onnx').stat().st_size / 1e6:.1f} MB)")
    else:
        with tempfile.TemporaryDirectory(prefix="shadowvox-hubertfa-") as tmp:
            archive = Path(tmp) / "1218_hfa_model_new_dict.zip"
            download(HUBERTFA_URL, archive, HUBERTFA_ARCHIVE_SHA256, force=True)
            print("  [UNZIP] 解包 HubertFA 中文对齐模型 …")
            _safe_extract_flat(archive, hfa_dir, "model.onnx")
        if not verify_file(hfa_dir / "model.onnx", HUBERTFA_ONNX_SHA256):
            raise SystemExit("解包后的 HubertFA model.onnx 校验失败")


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载并校验 ShadowVox 所需的全部离线模型。")
    parser.add_argument("--verify-only", action="store_true", help="仅校验本地模型文件，不下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载并覆盖")
    args = parser.parse_args(argv)

    if args.verify_only:
        checks = verify_all()
        all_ok = True
        for label, ok, detail in checks:
            mark = "OK" if ok else "MISS"
            print(f"  [{mark}] {label}: {detail}")
            all_ok = all_ok and ok
        return 0 if all_ok else 1

    fetch_all(force=args.force)
    print("全部模型就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
