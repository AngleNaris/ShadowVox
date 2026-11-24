"""ShadowVox 开发环境与外部依赖体检 / 一键补齐工具。

将项目移交给新开发者后，可通过本脚本快速核对哪些依赖已随仓库自带、哪些可通过
脚本自动拉取/构建、以及哪些需要手动安装系统软件。

用法:
  python tools/setup_env.py                  # 仅体检（等同 --check）
  python tools/setup_env.py --check          # 体检全部运行与打包依赖
  python tools/setup_env.py --install        # 自动补齐缺失的模型、ZL 插件与 Signalsmith 助手
  python tools/setup_env.py --install --all  # 同时安装 Python 依赖并构建 Python 3.12 降噪助手
"""
import argparse
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_denoise
import build_signalsmith
import fetch_models
import fetch_zl

REQUIRED_PACKAGES = (
    "numpy",
    "scipy",
    "soundfile",
    "onnxruntime-directml",
    "torch",
    "openai-whisper",
    "tiktoken",
    "pypinyin",
    "pedalboard",
    "praat-parselmouth",
    "numba",
    "PySide6-Essentials",
    "PySide6-Addons",
    "PySide6-WebEngine",
    "sounddevice",
    "pyinstaller",
)

OPTIONAL_PACKAGES = (
    "audio-separator",
)

VENDORED_IN_GIT = (
    ("HubertFA 推理子集", ROOT / "src/shadowvox/vendor/hubertfa/decoder.py"),
    ("Sync Toolbox 对齐子集", ROOT / "src/shadowvox/vendor/synctoolbox/dtw/mrmsdtw.py"),
    ("Signalsmith C++ 头文件", ROOT / "tools/signalsmith/third_party/stretch/signalsmith-stretch.h"),
)


def _find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return Path(found).resolve()
    for cand in (Path(r"C:\ffmpeg\bin\ffmpeg.exe"), Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")):
        if cand.is_file():
            return cand.resolve()
    return None


def _find_ffmpeg_license(ffmpeg_path):
    if not ffmpeg_path:
        return None
    for base in (ffmpeg_path.parent.parent, ffmpeg_path.parent):
        for name in ("LICENSE", "LICENSE.txt", "COPYING.GPLv3", "COPYING.LGPLv3", "COPYING"):
            cand = base / name
            if cand.is_file():
                return cand
    return None


def _has_msvc():
    try:
        build_signalsmith._compiler_setup()
        return True
    except RuntimeError:
        return False


def _print_row(status, category, name, detail):
    print(f"  [{status:4s}] [{category:6s}] {name}: {detail}")


def check_environment():
    """执行全面体检，返回缺失必需项数量。"""
    missing_required = 0

    print("=== 1. Python 运行时与依赖包 ===")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (sys.version_info.major, sys.version_info.minor) == (3, 13):
        _print_row("OK", "运行时", "主程序 Python", f"{py_ver} ({sys.executable})")
    else:
        _print_row("WARN", "运行时", "主程序 Python", f"当前 {py_ver}，推荐使用 Python 3.13（与打包配置一致）")

    for pkg in REQUIRED_PACKAGES:
        try:
            ver = md.version(pkg)
            extra = ""
            if pkg == "torch":
                if "+cu" in ver:
                    extra = " (注意: 检测到 CUDA 版 PyTorch，打包 EXE 体积会大幅膨胀，建议改用 CPU wheel)"
                elif "+cpu" in ver:
                    extra = " (CPU wheel，适合打包)"
            _print_row("OK", "Python", pkg, f"{ver}{extra}")
        except md.PackageNotFoundError:
            _print_row("MISS", "Python", pkg, "未安装 — 执行 pip install -r requirements-desktop.txt")
            missing_required += 1

    for pkg in OPTIONAL_PACKAGES:
        try:
            ver = md.version(pkg)
            _print_row("OK", "可选包", pkg, f"{ver}（仅用于非默认分离模型）")
        except md.PackageNotFoundError:
            _print_row("OPT", "可选包", pkg, "未安装（默认 Kim_Vocal_2 走便携 ONNX，不影响运行与打包）")

    print("\n=== 2. 仓库内置第三方源码 (Git 已包含) ===")
    for label, path in VENDORED_IN_GIT:
        if path.is_file():
            _print_row("OK", "已内置", label, path.relative_to(ROOT).as_posix())
        else:
            _print_row("MISS", "已内置", label, f"缺失 {path.relative_to(ROOT).as_posix()}")
            missing_required += 1

    print("\n=== 3. 系统外部工具 (需本机安装) ===")
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        lic = _find_ffmpeg_license(ffmpeg)
        lic_note = f"LICENSE: {lic.name}" if lic else "未找到同级 LICENSE（开发运行可用，PyInstaller 打包需 Gyan full build 或附带 LICENSE）"
        _print_row("OK" if lic else "WARN", "系统", "FFmpeg", f"{ffmpeg} ({lic_note})")
    else:
        _print_row("MISS", "系统", "FFmpeg", "未找到 — 请运行 winget install Gyan.FFmpeg 并重启终端")
        missing_required += 1

    if _has_msvc():
        _print_row("OK", "构建", "MSVC C++ 工具链", "已就绪（可编译 shadowvox-signalsmith.exe）")
    else:
        has_ss = build_signalsmith.OUTPUT.is_file()
        _print_row("WARN" if has_ss else "MISS", "构建", "MSVC C++ 工具链",
                   "未找到 — 编译 Signalsmith 需安装 Visual Studio Build Tools (C++ 桌面开发)")
        if not has_ss:
            missing_required += 1

    py312 = build_denoise.find_python312()
    if py312:
        _print_row("OK", "构建", "Python 3.12", f"{py312}（用于构建 shadowvox-denoise.exe）")
    else:
        has_dn, _ = build_denoise.check_ready()
        _print_row("OPT" if has_dn else "WARN", "构建", "Python 3.12",
                   "未找到 — 仅重建降噪助手时需要 (winget install Python.Python.3.12)")

    seven_z = shutil.which("7z")
    _print_row("OK" if seven_z else "OPT", "发布", "7-Zip (7z)",
               seven_z if seven_z else "未找到（仅制作 GitHub 分卷 Release 时可选）")

    print("\n=== 4. 离线 AI 模型 (可由 tools/fetch_models.py 自动补齐) ===")
    for label, ok, detail in fetch_models.verify_all():
        _print_row("OK" if ok else "MISS", "模型", label, detail)
        if not ok:
            missing_required += 1

    print("\n=== 5. VST3 插件 (可由 tools/fetch_zl.py 自动补齐) ===")
    for name, version, _, binary_sha in fetch_zl.RELEASES:
        bundle = f"ZL {name}.vst3"
        binary = ROOT / "plugins" / bundle / "Contents/x86_64-win" / bundle
        if binary.is_file() and fetch_models.sha256_file(binary) == binary_sha:
            _print_row("OK", "插件", f"ZL {name} {version}", f"{binary.stat().st_size / 1e6:.1f} MB")
        else:
            _print_row("MISS", "插件", f"ZL {name} {version}", "缺失或校验失败 — 运行 python tools/fetch_zl.py")
            missing_required += 1

    zl_lic = fetch_zl.LICENSE
    if zl_lic.is_file() and fetch_models.sha256_file(zl_lic) == fetch_zl.LICENSE_SHA256:
        _print_row("OK", "插件", "ZL AGPL-3.0 许可", zl_lic.relative_to(ROOT).as_posix())
    else:
        _print_row("MISS", "插件", "ZL AGPL-3.0 许可", "缺失 — 运行 python tools/fetch_zl.py")
        missing_required += 1

    print("\n=== 6. 原生/独立渲染与降噪助手 (可由脚本自动构建) ===")
    ss_exe = build_signalsmith.OUTPUT
    if ss_exe.is_file():
        _print_row("OK", "助手", "Signalsmith Stretch", f"{ss_exe.stat().st_size / 1e3:.0f} KB")
    else:
        _print_row("MISS", "助手", "Signalsmith Stretch", "缺失 — 运行 python tools/build_signalsmith.py")
        missing_required += 1

    dn_ok, dn_detail = build_denoise.check_ready()
    _print_row("OK" if dn_ok else "MISS", "助手", "DeepFilterNet3 降噪助手",
               dn_detail if dn_ok else f"{dn_detail} — 运行 python tools/build_denoise.py")
    if not dn_ok:
        missing_required += 1

    rb_exe = ROOT / "tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows/rubberband.exe"
    _print_row("OK" if rb_exe.is_file() else "OPT", "可选", "Rubber Band 4.0 CLI",
               "已安装（仅历史自然度对照脚本使用）" if rb_exe.is_file() else "未安装（默认引擎已切换为 Signalsmith，非必需）")

    print()
    if missing_required == 0:
        print("体检结果：全部运行与打包必需依赖均已就绪。")
    else:
        print(f"体检结果：共发现 {missing_required} 项必需依赖缺失或未通过校验。")
        print("提示：可执行 `python tools/setup_env.py --install` 自动拉取模型、ZL 插件并构建 Signalsmith；")
        print("      如需同时安装 Python 包并构建降噪助手，可执行 `python tools/setup_env.py --install --all`。")
    return missing_required


def install_dependencies(*, install_pip=False, build_denoise_helper=False, python312=None, force=False):
    if install_pip:
        print("=== [Step 1] 安装 Python 依赖 (含 CPU 版 PyTorch) ===")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                "torch",
            ],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu",
                "-r", str(ROOT / "requirements-desktop.txt"),
            ],
            cwd=ROOT, check=True,
        )

    print("\n=== [Step 2] 拉取并校验离线 AI 模型 (models/) ===")
    fetch_models.fetch_all(force=force)

    print("\n=== [Step 3] 拉取并校验 ZL VST3 插件 (plugins/) ===")
    fetch_zl.fetch()

    print("\n=== [Step 4] 构建 Signalsmith Stretch 原生助手 ===")
    if force or not build_signalsmith.OUTPUT.is_file():
        print("  构建结果:", build_signalsmith.build())
    else:
        print(f"  [OK] 已存在 {build_signalsmith.OUTPUT.relative_to(ROOT).as_posix()}")

    dn_ok, _ = build_denoise.check_ready()
    if build_denoise_helper or (not dn_ok and build_denoise.find_python312(python312) is not None):
        print("\n=== [Step 5] 构建 Python 3.12 DeepFilterNet3 降噪助手 ===")
        build_denoise.build(python312=python312, force=force)
    elif not dn_ok:
        print("\n=== [Step 5] 跳过 DeepFilterNet3 降噪助手（未检测到 Python 3.12）===")
        print("  请先安装 Python 3.12 后执行: python tools/build_denoise.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ShadowVox 依赖体检与一键补齐脚本。")
    parser.add_argument("--check", action="store_true", help="仅体检环境与依赖（默认动作）")
    parser.add_argument("--install", action="store_true", help="自动补齐模型、ZL 插件、Signalsmith（及可用的降噪助手）")
    parser.add_argument("--pip", action="store_true", help="与 --install 同用：同时安装 requirements-desktop.txt 与 CPU 版 PyTorch")
    parser.add_argument("--build-denoise", action="store_true", help="与 --install 同用：强制要求构建 Python 3.12 降噪助手")
    parser.add_argument("--all", action="store_true", help="与 --install 同用：等同同时指定 --pip 与 --build-denoise")
    parser.add_argument("--python312", help="指定 Python 3.12 路径（供构建降噪助手使用）")
    parser.add_argument("--force", action="store_true", help="强制重新下载/重建")
    args = parser.parse_args(argv)

    if args.install:
        install_dependencies(
            install_pip=args.pip or args.all,
            build_denoise_helper=args.build_denoise or args.all,
            python312=args.python312,
            force=args.force,
        )
        print("\n=== 补齐后重新体检 ===")
        missing = check_environment()
        return 0 if missing == 0 else 1

    missing = check_environment()
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
