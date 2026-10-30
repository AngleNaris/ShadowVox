"""用 Python 3.12 验证连续小块降噪与原整轨处理逐采样一致。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from deepfilternet_rs import DeepFilterNetRealtime

from denoise_helper import main


class DenoiseHelperTests(unittest.TestCase):
    def test_continuous_chunks_match_whole_track_including_tail(self):
        rng = np.random.default_rng(30)
        for length in (137, 48000 * 3 + 137):
            with self.subTest(samples=length), tempfile.TemporaryDirectory() as folder:
                t = np.arange(length) / 48000
                samples = (.2 * np.sin(2 * np.pi * (180 * t + 18 * t * t))
                           + .01 * rng.standard_normal(length)).astype(np.float32)
                source, target = Path(folder) / 'input.npy', Path(folder) / 'output.npy'
                np.save(source, samples)
                main(source, target)
                actual = np.load(target, allow_pickle=False)

                processor = DeepFilterNetRealtime(
                    atten_lim=12.0, compensate_delay=True,
                    post_filter_beta=0.0, log_level='error'
                )
                padded = np.pad(samples, (0, 1440))
                expected = np.concatenate((processor.process_chunk(padded),
                                           processor.finalize()))[:length]
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, samples.shape)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
