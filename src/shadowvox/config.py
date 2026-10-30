"""路径与模型常量。"""
from pathlib import Path
import os
import sys

ROOT = Path(sys._MEIPASS) if getattr(sys,"frozen",False) else Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA_ROOT = Path(os.environ.get("LOCALAPPDATA",str(Path.home()))) / "ShadowVox" if getattr(sys,"frozen",False) else ROOT
# 改名前的数据目录（分离缓存、VST3 设置）首次启动时整体迁移，避免重新分离整曲。
_LEGACY = DATA_ROOT.parent / "Atune"
if getattr(sys,"frozen",False) and _LEGACY.is_dir() and not DATA_ROOT.exists():
    try: _LEGACY.rename(DATA_ROOT)
    except OSError: pass
OUT_DIR = DATA_ROOT / "out"
WORK_DIR = DATA_ROOT / "work"

FCPE_ONNX = MODELS / "fcpe.onnx"
VOCODER_ONNX = MODELS / "hifigan_pkg" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
VOCAL_MDX_ONNX = MODELS / "mdx_vocals.onnx"

# FCPE 音高提取
F0_SR = 16000
F0_HOP = 160          # 100 fps
F0_FRAME_RATE = F0_SR / F0_HOP
F0_MIN = 32.7
F0_MAX = 1975.5
FCPE_CONF_THRESHOLD = 0.006

# 声码器
VOC_SR = 44100
VOC_HOP = 512
VOC_WIN = 2048
VOC_FFT = 2048
VOC_MELS = 128
VOC_FMIN = 40.0
VOC_FMAX = 16000.0
MEL_LOG_EPS = 1e-5
