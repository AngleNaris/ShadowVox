# ShadowVox 依赖清单与自动补齐说明

本文档面向接手开发的工程师，完整列出 ShadowVox 运行、调试与单文件打包所涉及的全部依赖，并明确标注：**哪些已随 Git 仓库自带**、**哪些可通过脚本自动下载/构建补齐**、**哪些需要目标开发机预装系统软件**，以及**离线交接打包方式**。

---

## 一、依赖体检与一键补齐

在任意新开发机上克隆仓库并创建 Python 3.13 虚拟环境后，可先执行依赖体检：

```powershell
.venv\Scripts\python.exe tools\setup_env.py --check
```

自动补齐所有可脚本化的外部依赖（离线模型、ZL VST3 插件、Signalsmith 原生助手；若本机已装 Python 3.12 也会自动构建降噪助手）：

```powershell
.venv\Scripts\python.exe tools\setup_env.py --install
```

若希望同时安装 Python 依赖（强制使用 CPU 版 PyTorch）并要求构建降噪助手：

```powershell
.venv\Scripts\python.exe tools\setup_env.py --install --all
```

---

## 二、依赖全景分类矩阵

### 1. 已包含在 Git 仓库内（无需额外获取）

| 组件 | 仓库路径 | 来源与版本 | 用途 | 许可证 |
| --- | --- | --- | --- | --- |
| HubertFA 推理子集 | `src/shadowvox/vendor/hubertfa/` | `wolfgitpr/HubertFA` (`b3f08696`) | 中文歌声拼音/音素强制对齐解码器 | Apache-2.0 |
| Sync Toolbox 子集 | `src/shadowvox/vendor/synctoolbox/` | `meinardmueller/synctoolbox` | MrMsDTW / 色度与 DLNCO 特征（历史及辅助对齐） | MIT |
| Signalsmith Stretch / Linear 头文件 | `tools/signalsmith/third_party/` | Stretch `1.3.2` / Linear `0.6.6` | 音高修正与受限变时 C++ 渲染引擎源码 | MIT |
| 界面前端静态资源 | `src/shadowvox/ui/` | 本项目 | 原生 HTML/CSS/JS 工作台与离线报告模板（无 npm 构建链） | — |
| 图标资源 | `assets/` | 本项目 | 窗口图标、EXE 图标与 README 图标 | — |

> 注意：`src/shadowvox/vendor/synctoolbox/` 如需重新从上游精简同步，可参考 `tools/vendor_synctoolbox.py`。

---

### 2. 可由脚本自动下载并校验（`.gitignore` 忽略，不入库）

#### 2.1 离线 AI 模型（`models/`，合计约 700 MB）

执行 `python tools/fetch_models.py` 自动下载全部模型，全部采用 `.part` 原子写入与 SHA-256 校验；仅校验本地文件可执行 `python tools/fetch_models.py --verify-only`。

| 本地路径 | 大小 | SHA-256 | 来源 | 用途 | 许可提醒 |
| --- | --- | --- | --- | --- | --- |
| `models/fcpe.onnx` | ~41.3 MB | `013b507a...9737e5d` | KCKT0112/HachiTune (CNChTu/FCPE) | 100 fps 音高提取 (F0) | MIT |
| `models/Kim_Vocal_2.onnx` | ~63.7 MB | `ce74ef3b...8cde9c8b` | TRvlvr/model_repo (UVR MDX-Net) | 原曲/干声人声与伴奏分离 | 权重许可未声明可任意商用 |
| `models/whisper-base.pt` | ~138.5 MB | `ed3a0b6b...26e34e` | OpenAI Whisper (`base`) | 本地离线唱词粗识别 | MIT |
| `models/hubertfa/` (`model.onnx` + 词典/配置) | ~419 MB (压缩包 ~257 MB) | 压缩包 `48bd6dbc...f346fa2`<br>模型 `4722c13e...be170183` | `wolfgitpr/HubertFA` `v0.0.7` (`1218_hfa_model_new_dict.zip`) | 中文音素级强制对齐 | 代码 Apache-2.0；官方权重见 `THIRD_PARTY.txt` |
| `models/hifigan_pkg/` (`pc_nsf_hifigan_...onnx` + `*.txt`) | ~54.0 MB (压缩包 ~50.3 MB) | 压缩包 `ba7d4314...e6902e`<br>模型 `b297c0c7...686155d` | OpenVPI `pc-nsf-hifigan-44.1k-hop512-128bin-2025.02` | 历史/对照声码器渲染（`ShadowVox.spec` 仍打包） | **CC BY-NC-SA 4.0（非商业限制）** |

> 清理提示：如果老工作区 `models/` 中存在 `mdx_vocals.onnx`、`mdx_model_data.json`、`vr_model_data.json`、`download_checks.json`，它们属于早期 `audio-separator` 或旧版拉取脚本遗留；当前默认链路仅读取上述 5 项。新版 `tools/fetch_models.py` 会自动把已校验通过的 `mdx_vocals.onnx` 复用复制到 `Kim_Vocal_2.onnx`。

#### 2.2 VST3 音频插件（`plugins/`，合计约 14 MB）

执行 `python tools/fetch_zl.py` 自动从 GitHub Releases 下载 MSI、通过 `msiexec /a` 解包提取 `.vst3`、下载 AGPL-3.0 许可文本并逐项校验 SHA-256。

| 本地路径 | 版本 | 用途 | 许可证 |
| --- | --- | --- | --- |
| `plugins/ZL Compressor.vst3` | `0.6.1` | 四段 Optical 母带密度压缩 (`density.py`) | AGPL-3.0 |
| `plugins/ZL Limiter.vst3` | `0.0.1` | 4x 过采样真峰值母带限幅 (`mastering.py`) | AGPL-3.0 |
| `plugins/ZL-AGPL-3.0.txt` | — | 随单文件 EXE 打包的许可声明 | AGPL-3.0 |

---

### 3. 可由脚本自动构建的独立子程序（`tools/vendor/`，`.gitignore` 忽略）

| 产物路径 | 构建命令 | 构建前置条件 | 说明 |
| --- | --- | --- | --- |
| `tools/vendor/signalsmith/shadowvox-signalsmith.exe` | `python tools/build_signalsmith.py` | Visual Studio Build Tools（含 C++ 桌面开发 / `cl.exe`） | 源码已随仓库内置于 `tools/signalsmith/`，构建仅需数秒。`build_release.py` 打包正式版时也会自动重编。 |
| `tools/vendor/deepfilternet/shadowvox-denoise.exe`<br>`tools/vendor/deepfilternet/licenses/` | `python tools/build_denoise.py` | **Python 3.12**（Windows x64） | `deepfilternet-rs 0.1.1` 仅支持 Python 3.10–3.12，因此单独用 Python 3.12 + `requirements-denoise-helper.txt` + `tools/denoise_helper.spec` 打包成独立 EXE，由 Python 3.13 主程序按子进程调用。脚本会自动收集全部随包许可文件。 |

---

### 4. Python 依赖包（由 `pip` 自动安装）

主程序和桌面打包统一使用 **Python 3.13（Windows x64）**，清单见 [requirements.txt](../requirements.txt) 与 [requirements-desktop.txt](../requirements-desktop.txt)。

**强烈建议安装命令（强制使用 CPU 版 PyTorch）：**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-desktop.txt
```

**关键说明：**

1. **为什么必须使用 CPU 版 PyTorch**：Whisper 在本机固定使用 `device='cpu'` 推理（见 `hubert_alignment.py` 与 `lyric_anchors.py`），GPU 加速仅用于 ONNX Runtime（`onnxruntime-directml`）。若误装默认 CUDA 版 PyTorch，单文件 EXE 会从约 1.1 GB 暴增至 3 GB 以上。
2. **哪些包不会被打包进单文件 EXE**：`ShadowVox.spec` 明确排除了 `torchvision`、`torchaudio`、`audio_separator`、`onnx2torch`、`onnx`、`librosa`、`sklearn`、`matplotlib`、`pandas`、`synctoolbox`、`libfmp` 等重型包。默认分离路径（`Kim_Vocal_2.onnx`）使用轻量 `mdx_portable.py`（纯 NumPy/SciPy + ONNX Runtime），只有 CLI 手动传 `--sep-model <其他模型>` 时才延迟导入 `audio-separator`。
3. **测试/实验工具额外包**：运行研究对比脚本（如 `tools/plot_rhythm.py`、`tools/make_icon.py`）可能需要 `matplotlib` 或 `pillow`（未放入 `requirements-desktop.txt` 主运行时，需要时手动 `pip install matplotlib pillow`）。

---

### 5. 必须由开发机预装的系统级工具（不可完全由 Python 自行安装）

| 工具 | 必需阶段 | 推荐安装命令（Windows PowerShell） | 注意事项 |
| --- | --- | --- | --- |
| **Python 3.13 (x64)** | 运行、开发、打包 | `winget install Python.Python.3.13` | 主程序运行环境与 `ShadowVox.spec` 打包环境 |
| **FFmpeg (Gyan full build)** | 运行、开发、打包 | `winget install Gyan.FFmpeg` | 音频统一由 `ffmpeg` 解码、削波修复与响度测量。**打包时 `ShadowVox.spec` 会收集 FFmpeg 同级或上级目录的 `LICENSE` 文件**，推荐使用 Gyan 官方构建。 |
| **Visual Studio Build Tools (C++ 桌面开发)** | 编译 Signalsmith 助手 | 安装 Visual Studio Installer 并勾选「使用 C++ 的桌面开发」(`VC.Tools.x86.x64`) | `tools/build_signalsmith.py` 通过 `vswhere.exe` 自动定位 `vcvars64.bat` |
| **Python 3.12 (x64)** | 仅重建降噪助手时需要 | `winget install Python.Python.3.12` | 若通过离线依赖包直接拷贝 `tools/vendor/deepfilternet/`，则无需在开发机安装 Python 3.12 |
| **7-Zip (`7z`)** | 可选（仅发布分卷压缩） | `winget install 7zip.7zip` | GitHub 单文件上传限制 2 GB / 分卷分发用 |

---

### 6. 可选 / 历史研究依赖（默认功能不需要）

| 组件 | 路径 | 说明 |
| --- | --- | --- |
| Rubber Band 4.0.0 CLI | `tools/vendor/rubberband-4.0.0/rubberband-4.0.0-gpl-executable-windows/` | 仅 `src/shadowvox/rubberband.py` 及历史渲染对照脚本（`tools/renderer_comparisons.py`、`tools/naturalness_lab.py`）使用；桌面和 CLI 默认渲染器自 9.5.1 起已全面切换为 Signalsmith。 |
| 商业 VST3 插件（如 iZotope RX） | 系统 VST3 目录 | 仅 `tools/test_vst_gui.py` 测试原生 VST3 窗口打开/状态恢复时引用了作者本机路径。 |

---

## 三、离线 / 受限网络交接方案

如果接收方处于内网环境或访问 GitHub Releases / Azure 较慢，原开发者除移交 Git 仓库外，只需将以下 **3 个被 `.gitignore` 忽略的目录**打包为一份离线依赖包随源码一并移交（解压到仓库根目录即可，总计约 750 MB）：

```text
models/
├── fcpe.onnx
├── Kim_Vocal_2.onnx
├── whisper-base.pt
├── hifigan_pkg/
│   ├── pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx
│   ├── NOTICE.txt / NOTICE.zh-CN.txt / STATEMENTS.txt / ...
└── hubertfa/
    ├── model.onnx
    ├── config.json / vocab.json / VERSION
    └── ds-zh-pinyin-lite.txt / ds_cmudict-07b.txt / japanese_dict_full.txt

plugins/
├── ZL Compressor.vst3/
├── ZL Limiter.vst3/
└── ZL-AGPL-3.0.txt

tools/vendor/
├── deepfilternet/
│   ├── shadowvox-denoise.exe
│   └── licenses/
└── signalsmith/
    └── shadowvox-signalsmith.exe
```

接收方解压后运行一次体检脚本即可确认所有文件和 SHA-256 完整无误：

```powershell
.venv\Scripts\python.exe tools\setup_env.py --check
```
