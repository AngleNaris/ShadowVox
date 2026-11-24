"""使用 Python 3.12 构建独立 AI 降噪子程序 (shadowvox-denoise.exe) 并收集随包许可。

deepfilternet-rs 0.1.1 仅提供 Python 3.10-3.12 的预编译包，因此降噪助手单独用
Python 3.12 打包为独立 EXE，由主程序（Python 3.13）通过子进程调用。

用法:
  python tools/build_denoise.py
  python tools/build_denoise.py --python C:\\path\\to\\python312\\python.exe
  python tools/build_denoise.py --check
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "tools" / "vendor" / "deepfilternet"
OUTPUT_EXE = VENDOR_DIR / "shadowvox-denoise.exe"
LICENSES_DIR = VENDOR_DIR / "licenses"
SPEC_FILE = ROOT / "tools" / "denoise_helper.spec"
REQ_FILE = ROOT / "requirements-denoise-helper.txt"
DEFAULT_VENV = ROOT / "out" / "build-denoise-venv"

UPSTREAM_MIT_URL = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/LICENSE-MIT"

_COLLECT_LICENSES_SNIPPET = r"""
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

# 1. Python 3.12 许可
py_lic = Path(sys.base_prefix) / "LICENSE.txt"
if py_lic.is_file():
    shutil.copy2(py_lic, out / "python-3.12.txt")

# 2. deepfilternet-rs 元数据与 SBOM
df_dist = md.distribution("deepfilternet-rs")
meta_path = df_dist._path / "METADATA"
if meta_path.is_file():
    shutil.copy2(meta_path, out / "deepfilternet-rs-metadata.txt")
for rel in df_dist.files or []:
    name = str(rel).lower()
    if name.endswith("cyclonedx.json"):
        src = Path(df_dist.locate_file(rel))
        if src.is_file():
            shutil.copy2(src, out / "deepfilternet-rs.cyclonedx.json")

# 3. NumPy 许可文件
np_out = out / "numpy"
if np_out.exists():
    shutil.rmtree(np_out)
np_dist = md.distribution("numpy")
for rel in np_dist.files or []:
    upper = str(rel).upper()
    if "LICENSE" in upper or "COPYING" in upper:
        src = Path(np_dist.locate_file(rel))
        if src.is_file():
            # 保持与原有 licenses/numpy 目录结构一致
            rel_path = Path(str(rel))
            if rel_path.parts and rel_path.parts[0].endswith(".dist-info"):
                dest = np_out / rel_path.name
            else:
                dest = np_out / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
"""


def find_python312(explicit=None):
    """定位可用的 Python 3.12 解释器。"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        py_launcher = shutil.which("py")
        if py_launcher:
            proc = subprocess.run(
                [py_launcher, "-3.12", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, encoding="utf-8",
            )
            if proc.returncode == 0 and proc.stdout.strip():
                candidates.append(Path(proc.stdout.strip()))
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.extend([
            local / "Programs" / "Python" / "Python312" / "python.exe",
            Path(r"C:\Python312\python.exe"),
            Path(r"C:\Program Files\Python312\python.exe"),
        ])

    for cand in candidates:
        if not cand.is_file():
            continue
        proc = subprocess.run(
            [str(cand), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if proc.returncode == 0 and proc.stdout.strip() == "3.12":
            return cand
    return None


def check_ready():
    """检查降噪助手二进制与许可目录是否就绪，返回 (ok, detail)。"""
    if not OUTPUT_EXE.is_file():
        return False, f"缺少 {OUTPUT_EXE.relative_to(ROOT).as_posix()}"
    required_licenses = (
        LICENSES_DIR / "DeepFilterNet-LICENSE-MIT.txt",
        LICENSES_DIR / "deepfilternet-rs-metadata.txt",
        LICENSES_DIR / "python-3.12.txt",
    )
    missing = [p.name for p in required_licenses if not p.is_file()]
    if missing:
        return False, f"许可目录不完整，缺少: {', '.join(missing)}"
    return True, f"{OUTPUT_EXE.stat().st_size / 1e6:.1f} MB，许可目录完整"


def build(*, python312=None, venv_dir=DEFAULT_VENV, force=False):
    if not force:
        ok, detail = check_ready()
        if ok:
            print(f"[OK] shadowvox-denoise.exe 已就绪 ({detail})")
            return OUTPUT_EXE

    py312 = find_python312(python312)
    if py312 is None:
        raise RuntimeError(
            "未找到 Python 3.12（deepfilternet-rs 0.1.1 仅支持 Python 3.10–3.12）。\n"
            "请先安装 Python 3.12（例如: winget install Python.Python.3.12），\n"
            "或通过 --python 指定 python3.12 可执行文件路径。"
        )
    print(f"[1/4] 使用 Python 3.12: {py312}")

    venv_dir = Path(venv_dir)
    venv_python = venv_dir / "Scripts" / "python.exe"
    if not venv_python.is_file():
        print(f"[2/4] 创建构建虚拟环境: {venv_dir}")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(py312), "-m", "venv", str(venv_dir)], check=True)
    else:
        print(f"[2/4] 复用构建虚拟环境: {venv_dir}")

    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)],
        check=True,
    )

    print("[3/4] 使用 PyInstaller 打包 shadowvox-denoise.exe …")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shadowvox-denoise-build-") as work:
        subprocess.run(
            [
                str(venv_python), "-m", "PyInstaller", "--noconfirm", "--clean",
                "--distpath", str(VENDOR_DIR),
                "--workpath", str(work),
                str(SPEC_FILE),
            ],
            cwd=ROOT, check=True,
        )

    if not OUTPUT_EXE.is_file():
        raise RuntimeError(f"打包完成但未找到输出文件: {OUTPUT_EXE}")

    print("[4/4] 收集第三方许可文件到 tools/vendor/deepfilternet/licenses/ …")
    LICENSES_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(venv_python), "-c", _COLLECT_LICENSES_SNIPPET, str(LICENSES_DIR)],
        check=True,
    )
    mit_path = LICENSES_DIR / "DeepFilterNet-LICENSE-MIT.txt"
    if not mit_path.is_file():
        try:
            req = Request(UPSTREAM_MIT_URL, headers={"User-Agent": "ShadowVox-Build"})
            with urlopen(req, timeout=15) as resp:
                mit_path.write_bytes(resp.read())
        except Exception as exc:
            print(f"[WARN] 下载 DeepFilterNet 上游 LICENSE-MIT 失败: {exc}", file=sys.stderr)

    print(f"构建完成: {OUTPUT_EXE} ({OUTPUT_EXE.stat().st_size / 1e6:.1f} MB)")
    return OUTPUT_EXE


def main(argv=None):
    parser = argparse.ArgumentParser(description="构建 DeepFilterNet3 独立降噪助手及许可目录。")
    parser.add_argument("--python", help="指定 Python 3.12 可执行文件路径")
    parser.add_argument("--venv", default=str(DEFAULT_VENV), help="构建用 Python 3.12 虚拟环境路径")
    parser.add_argument("--force", action="store_true", help="即使已存在也强制重新构建")
    parser.add_argument("--check", action="store_true", help="仅检查降噪助手与许可是否就绪")
    args = parser.parse_args(argv)

    if args.check:
        ok, detail = check_ready()
        print(f"  [{'OK' if ok else 'MISS'}] DeepFilterNet3 降噪助手: {detail}")
        return 0 if ok else 1

    try:
        build(python312=args.python, venv_dir=args.venv, force=args.force)
    except RuntimeError as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
