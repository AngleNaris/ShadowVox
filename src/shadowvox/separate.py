"""原曲人声分离（MDX-Net / UVR 模型），拿原唱人声当音高参考，伴奏用于成品混音。"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import audio, config as C

DEFAULT_MODEL = "Kim_Vocal_2.onnx"


def _use_directml():
    return "DmlExecutionProvider" in ort.get_available_providers()


def separate_song(song_path, work_dir, model=DEFAULT_MODEL, log=print):
    """返回 dict(vocals=[n] mono 44.1k, instrumental=[2,n] 44.1k, paths...)。"""
    if model == DEFAULT_MODEL:
        from .mdx_portable import separate
        return separate(song_path,log=log)
    from audio_separator.separator import Separator

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sep = Separator(
        log_level=30,
        model_file_dir=str(C.MODELS),
        output_dir=str(work_dir),
        output_format="wav",
        use_directml=_use_directml(),
        # libsndfile 写 32-bit float WAV 会报 "unsupported encoding"，
        # 关掉 soundfile 走 ffmpeg 写盘（项目本来就依赖 ffmpeg）。
        use_soundfile=False,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": False,
        },
    )
    sep.load_model(model_filename=model)
    log(f"[分离] 模型 {model}，设备 {'DirectML' if _use_directml() else 'CPU'}")
    outputs = sep.separate(str(song_path))

    # 注意：Instrumental 的文件名里也含 "Vocal"（Kim_Vocal_2），不能只查子串，
    # 否则会把伴奏当人声。按 (Vocals) / (Instrumental) 标签严格判断。
    buckets = "".join(Path(p).name for p in outputs)
    log(f"[分离] 产出 {len(outputs)} 轨：{buckets}")

    def pick(tag):
        want = f"({tag})"
        for p in outputs:
            if want in Path(p).name:
                return p
        return None

    vocal_path = pick("Vocals")
    inst_path = pick("Instrumental")
    # 兜底：按“不含 Instrumental”过滤后取唯一一个
    if vocal_path is None:
        rest = [p for p in outputs if "Instrumental" not in Path(p).name]
        vocal_path = rest[0] if len(rest) == 1 else None
    if vocal_path is None:
        raise RuntimeError(f"分离未产出人声轨: {outputs}")

    vocals = audio.load_mono(Path(work_dir) / vocal_path, C.VOC_SR)
    instrumental = (audio.load(Path(work_dir) / inst_path, sr=C.VOC_SR, channels=2)[0]
                    if inst_path else None)

    # 自检：人声轨与伴奏轨若高度相关，说明取错了文件，必须拦下
    if instrumental is not None:
        n = min(len(vocals), instrumental.shape[1])
        if n > C.VOC_SR and np.std(vocals[:n]) > 1e-9:
            corr = abs(float(np.corrcoef(vocals[:n], instrumental.mean(axis=0)[:n])[0, 1]))
            if corr > 0.9:
                raise RuntimeError(
                    f"人声轨与伴奏轨近乎相同（相关系数 {corr:.3f}），分离输出异常：{outputs}")

    return {
        "vocals": vocals,
        "instrumental": instrumental,
        "vocal_path": Path(work_dir) / vocal_path,
        "inst_path": Path(work_dir) / inst_path if inst_path else None,
    }
