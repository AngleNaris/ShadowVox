"""面板参数、局部试听和报告的针对性回归。"""
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from shadowvox import mixing
from shadowvox.full_align import plan_segments
from shadowvox.live_preview import PreviewCache
from shadowvox.pipeline import Options
from shadowvox.reporting import build, html_report, spectrum, summary_rows


class AudioControlsTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.arange(2*self.sr)/self.sr
        envelope = np.where(t < 1, .35, .10)
        self.vocal = (envelope*(np.sin(2*np.pi*180*t)+.8*np.sin(2*np.pi*5500*t))).astype(np.float32)
        self.bed = np.stack([.15*np.sin(2*np.pi*440*t), .15*np.sin(2*np.pi*880*t)]).astype(np.float32)

    def test_each_mix_control_changes_the_signal(self):
        original, _ = mixing.enhance(self.vocal, self.bed, self.sr)
        for key, value in (('lowcut_hz', 0), ('eq', 0), ('compression', 0), ('deessing', 0),
                           ('reverb', 0), ('delay_amount', 0), ('duck', 0)):
            with self.subTest(control=key):
                changed, info = mixing.enhance(self.vocal, self.bed, self.sr, **{key: value})
                self.assertGreater(float(np.max(abs(original-changed))), 1e-5)
                self.assertTrue(np.isfinite(changed).all())
        _, info = mixing.enhance(self.vocal, self.bed, self.sr, eq=0, duck=0, reverb=0, delay_amount=0)
        self.assertEqual(info['eq_bands'][0][1], 0)
        self.assertEqual(info['max_band_duck_db'], 0)
        self.assertEqual(info['reverb_wet'], 0)
        self.assertEqual(info['delay_wet'], 0)

    def test_rhythm_amount_interpolates_and_preserves_protected_runs(self):
        dry = np.arange(0., 10.01, .1)
        ref = dry + .12*np.sin(np.pi*dry/10)
        args = (dry, ref, 10., 12., np.ones(100, bool))
        evidence = np.ones(100, bool)
        full, _ = plan_segments(*args, acoustic_trusted=evidence)
        half, _ = plan_segments(*args, strength=.5, acoustic_trusted=evidence)
        self.assertAlmostEqual(full[-1]['target'][-1], 10., delta=.01)
        self.assertGreater(np.max(full[0]['target']-full[0]['source']), .01)
        np.testing.assert_allclose(half[0]['target']-half[0]['source'],
                                   (full[0]['target']-full[0]['source'])*.5, atol=1e-8)
        protected, stats = plan_segments(*args, anchors=[], strength=.5)
        self.assertGreater(stats['protected_seconds'], 9.9)
        for piece in protected:
            np.testing.assert_allclose(np.diff(piece['source']), np.diff(piece['target']), atol=1e-8)

    def test_denoise_dial_uses_actual_cleaned_signal(self):
        raw = np.ones(100, dtype=np.float32)
        clean = raw * .5
        capture = dict(options={'declip': True}, dry_input=raw, dry_before_denoise=raw,
                       denoised_dry=clean)
        cache = PreviewCache(capture, {}, {'gains_db': [0, 0]})
        np.testing.assert_allclose(cache.source(Options(denoise_strength=0)), raw)
        np.testing.assert_allclose(cache.source(Options(denoise_strength=.5)), raw*.75)
        np.testing.assert_allclose(cache.source(Options(denoise_strength=1)), clean)

    def test_draft_reuses_tracks_and_updates_pitch_rhythm_and_denoise(self):
        sr = 44100
        t = np.arange(8*sr)/sr
        raw = (.18*np.sin(2*np.pi*210*t)).astype(np.float32)
        clean = raw*.8
        bed = np.stack([.03*np.sin(2*np.pi*440*t)]*2).astype(np.float32)
        baseline = Options(strength=.75, rhythm=True, rhythm_mode='full', denoise=True,
                           mix_effects=True, renderer='psola')
        times = np.arange(0., 8.01, .1)
        capture = dict(options=asdict(baseline), dry_input=raw, dry_before_denoise=raw,
            denoised_dry=clean, aligned_dry=clean, vocal=clean, inst=bed, inst_original=bed,
            reference_original=raw, vocal_normalization=1., move=0.,
            shift_cents=np.full(801, 30., dtype=np.float32),
            rhythm_plan=dict(dry_t=times, ref_t=times*1.1, dry_seconds=8., ref_seconds=8.8,
                voiced=np.ones(80, bool), anchors=None, offset=0., acoustic_trusted=None))
        cache = PreviewCache(capture, {'mixing': {'master_gain_db': 0, 'target_lufs': -16}},
                             {'gains_db': [0, 0]})
        _, normal = cache.render(baseline, 1., seconds=4)
        for opts in (replace(baseline, mix_eq=1.5), replace(baseline, strength=.5),
                     replace(baseline, rhythm_strength=.5, strength=0),
                     replace(baseline, denoise_strength=.5, strength=0)):
            with self.subTest(options=opts):
                _, changed = cache.render(opts, 1., seconds=4)
                self.assertEqual(changed.shape, normal.shape)
                self.assertTrue(np.isfinite(changed).all())
                self.assertGreater(float(np.max(abs(changed-normal))), 1e-5)

    def test_master_targets_are_measured_from_actual_output(self):
        result, stats = mixing.master(self.bed, self.sr, target_lufs=-20, true_peak_ceiling=-2.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'out.wav'; sf.write(path, result.T, self.sr)
            lufs, peak = mixing.measure_loudness(path)
        self.assertAlmostEqual(lufs, stats['estimated_output_lufs'], delta=.2)
        self.assertLessEqual(peak, -1.95)
        self.assertEqual(stats['target_lufs'], -20)


class ReportTests(unittest.TestCase):
    def test_report_measures_audio_and_escapes_filename(self):
        sr = 16000
        t = np.arange(4*sr)/sr
        before = np.stack([.1*np.sin(2*np.pi*1000*t)]*2, axis=1)
        after = before * .5
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'before.wav', Path(folder)/'out.wav'
            sf.write(a, before, sr); sf.write(b, after, sr)
            report = {'options': {'strength': .75, 'rhythm': False}, 'output': str(b),
                      'denoise': {'enabled': False}, 'declip': {'applied': False}}
            metrics = build(report, a, b)
            html = html_report(report, metrics, '<output>.wav')
        self.assertAlmostEqual(metrics['before']['lufs'] - metrics['output']['lufs'], 6.02, delta=.1)
        self.assertAlmostEqual(metrics['before']['true_peak_dbtp'] - metrics['output']['true_peak_dbtp'], 6.02, delta=.1)
        self.assertEqual(metrics['output']['seconds'], 4.)
        peak_index = int(np.argmax(metrics['spectrum']['before']))
        self.assertEqual(metrics['spectrum']['centers'][peak_index], 1000)
        self.assertIn('&lt;output&gt;.wav', html)
        self.assertIn('<svg', html)

    def test_stereo_spectrum_does_not_cancel_opposite_channels(self):
        sr = 16000
        t = np.arange(2*sr)/sr
        tone = .1*np.sin(2*np.pi*1000*t)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'stereo.wav'
            sf.write(path, np.stack([tone, -tone], axis=1), sr)
            values = spectrum(path)
        self.assertEqual(int(np.argmax(values)), 7)
        self.assertGreater(values[7], -40)

    def test_protected_rhythm_intervals_include_listening_positions(self):
        report = {'options': {'strength': .75, 'rhythm': True},
                  'rhythm': {'protected_seconds': 2., 'protected_intervals_seconds':
                             [[.4, .7], [.7, 1.1], [61.2, 62.5]]}}
        rows = dict(summary_rows(report, {}))
        self.assertEqual(rows['建议试听位置'], '00:00.4–00:01.1、01:01.2–01:02.5')


class DesktopBridgeTests(unittest.TestCase):
    """界面交互在 tools/test_gui.py 用真实窗口验证；这里只测桥接层的纯逻辑。"""
    DEFAULTS = dict(strength=75, rhythm_strength=100, denoise_strength=100, vocal_gain_db=0, target_lufs=-16,
                    true_peak_ceiling=-1.5, mix_lowcut_hz=75, mix_eq=100, mix_compression=100, mix_deess=100,
                    mix_reverb=100, mix_delay=100, mix_duck=100, rhythm=True, denoise=True, declip=True, mix_effects=True)

    def test_front_end_defaults_map_to_confirmed_options(self):
        from shadowvox.gui_bridge import options_from
        o = options_from(self.DEFAULTS)
        self.assertEqual((o.strength, o.pitch_mode, o.renderer, o.rhythm_mode, o.max_shift_cents), (.75, 'register', 'psola', 'full', 450))
        self.assertEqual((o.mix_eq, o.mix_reverb, o.mix_lowcut_hz, o.target_lufs, o.true_peak_ceiling, o.vocal_gain_db), (1., 1., 75., -8., -.5, 0.))
        self.assertTrue(o.mix_effects and o.rhythm and o.denoise and o.declip and o.comparison)
        o = options_from({**self.DEFAULTS, 'mix_eq': 150, 'vocal_gain_db': -2.5, 'denoise': False})
        self.assertEqual((o.mix_eq, o.vocal_gain_db, o.denoise), (1.5, -2.5, False))
        for mixdown in (0, 100):
            o = options_from(dict(correction=50, beauty=50, fusion=50, ratio=35, mixdown=mixdown,
                                  mix_effects=False, target_lufs=-24, true_peak_ceiling=-4))
            self.assertEqual((o.fixed_master, o.target_lufs, o.true_peak_ceiling, o.master_density),
                             (True, -8., -.5, mixdown/100))

    def test_versions_never_overwrite(self):
        from shadowvox.jobs import next_version
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(next_version(folder, '干声'), 1)
            for name in ('干声_V1.wav', '干声_V3.wav', '干声_V2.report.json', '其他_V9.wav'):
                (Path(folder)/name).write_bytes(b'')
            self.assertEqual(next_version(folder, '干声'), 4)

    def test_player_loop_wraps_and_delta_subtracts_before(self):
        from types import SimpleNamespace
        from shadowvox.player import ComparisonPlayer
        player = ComparisonPlayer(); player.volume = 1.
        ramp = np.repeat(np.arange(100, dtype=np.float32)[:, None], 2, axis=1)
        player.tracks, player.sr = [ramp, ramp*2, ramp*3], 100
        player.set_loop(.2, .3); player.seek(.25); player.playing = True
        out = np.zeros((20, 2), dtype=np.float32)
        player._callback(out, 20, None, SimpleNamespace(output_underflow=False))
        np.testing.assert_array_equal(out[:, 0], [25, 26, 27, 28, 29, *range(20, 30), 20, 21, 22, 23, 24])
        player.loop = None; player.select(2); player.delta = True; player.seek(.5)
        player._callback(out, 4, None, SimpleNamespace(output_underflow=False))
        np.testing.assert_array_equal(out[:4, 0], [100, 102, 104, 106])   # 3x - x
        player.delta = False; player.seek(.98)
        player._callback(out, 4, None, SimpleNamespace(output_underflow=False))
        self.assertFalse(player.playing); np.testing.assert_array_equal(out[2:4, 0], [0, 0])

    def test_waveform_uses_stereo_energy_and_includes_tail(self):
        from shadowvox.gui_bridge import envelope, wave_values
        track = np.array([[.3, 0], [.3, 0], [0, 0], [0, 0], [0, -.9]], dtype=np.float32)
        peak, rms = envelope(track, 2)
        np.testing.assert_allclose(peak, [.3, .9])
        np.testing.assert_allclose(rms, [.3/np.sqrt(2), .9/np.sqrt(6)])
        values = wave_values(*envelope(track*.25), gain=4)
        self.assertEqual(values['peak'], [.3, .3, 0., 0., .9])


if __name__ == '__main__':
    unittest.main()
