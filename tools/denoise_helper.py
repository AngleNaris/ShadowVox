"""DeepFilterNet3 的独立进程入口；输入和输出均为 48 kHz 单声道 float32 NPY。"""
import sys

import numpy as np
from deepfilternet_rs import DeepFilterNetRealtime


def main(input_path, output_path):
    samples = np.asarray(np.load(input_path, allow_pickle=False), dtype=np.float32)
    processor = DeepFilterNetRealtime(
        atten_lim=12.0, compensate_delay=True, post_filter_beta=0.0, log_level="error"
    )
    # 补足模型的 30 ms 延迟，让末尾内容也能通过模型，然后截回原长度。
    padded = np.pad(samples, (0, 1440))
    # 实时后端每帧都会搬移剩余输入；限制为 1 秒小块，避免整轨处理的平方级开销。
    # 共用同一个 processor，分块之间保留模型状态，仅在全曲末尾 finalize。
    chunks = [processor.process_chunk(padded[start:start + 48000])
              for start in range(0, len(padded), 48000)]
    output = np.concatenate((*chunks, processor.finalize()))
    np.save(output_path, np.asarray(output[:len(samples)], dtype=np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
