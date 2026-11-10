"""Batch isolation, desktop CLI parameters and GitHub update responses."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shadowvox import __version__, cli, jobs, pipeline, updates


class JobTests(unittest.TestCase):
    def test_batch_continues_and_applies_per_item_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            song = folder/'song.wav'
            song.touch()
            dry = [folder/f'dry{i}.wav' for i in range(3)]
            dry[0].touch()
            dry[2].touch()
            items = [dict(dry=p, song=song, dry_has_backing=i == 2) for i, p in enumerate(dry)]
            statuses = []
            with patch.object(jobs, 'export_job', return_value={'path': 'mixed.wav', 'html': 'report.html'}) as export:
                results = jobs.run_batch(items, folder, jobs.DEFAULT_PARAMS, update=statuses.append)
                self.assertEqual([c.args[3].dry_has_backing for c in export.call_args_list], [False, True])
                self.assertTrue(all(c.args[3].fixed_master for c in export.call_args_list))
            self.assertEqual([r['state'] for r in results], ['done', 'error', 'done'])
            self.assertEqual([r['state'] for r in statuses], ['running', 'done', 'running', 'error', 'running', 'done'])

    def test_stop_does_not_start_following_items(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)/'input.wav'
            audio.touch()
            states = []
            with patch.object(jobs, 'export_job', side_effect=lambda *args, log, **kw: log('下一阶段')) as export:
                results = jobs.run_batch([dict(dry=audio, song=audio)]*2, directory, jobs.DEFAULT_PARAMS,
                                         update=states.append, cancelled=lambda: bool(states))
            self.assertEqual(results, [dict(index=0, state='cancelled')])
            self.assertEqual(export.call_count, 1)

    def test_new_reference_cache_keeps_first_run_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            song = folder/'song.wav'
            song.touch()
            vocal = np.linspace(-.1, .1, 200, dtype=np.float32)
            accompaniment = np.stack((vocal*12, vocal*13))
            stems = dict(vocals=vocal, instrumental=accompaniment)
            with patch.object(pipeline, 'separate_song', return_value=stems) as separate, \
                 patch.object(pipeline, '_stems_ok', return_value=True):
                first = pipeline._load_or_separate(song, folder, None, True, lambda _: None)
                cached = pipeline._load_or_separate(song, folder, None, True, lambda _: None)
            self.assertEqual(separate.call_count, 1)
            for a, b in zip(first, cached):
                np.testing.assert_array_equal(a, b)

    def test_cli_manifest_and_desktop_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/'items.json'
            manifest.write_text('[{"dry":"vocal.wav","song":"reference.wav","dry_has_backing":true}]', encoding='utf-8')
            with patch.object(jobs, 'run_batch', return_value=[]) as batch:
                self.assertEqual(cli.main(['batch', '--manifest', str(manifest), '-o', directory,
                                           '--ratio', '-8', '--mixdown', '65', '--no-post']), 0)
            items, output, params = batch.call_args.args
            self.assertEqual(items[0]['dry'], str(manifest.parent/'vocal.wav'))
            self.assertTrue(items[0]['dry_has_backing'])
            self.assertEqual(output, manifest.parent)
            self.assertEqual((params['ratio'], params['mixdown'], params['mix_effects']), (-8, 65, False))


class UpdateTests(unittest.TestCase):
    def check_release(self, data):
        with patch.object(updates, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())) as request:
            result = updates.check_update()
            self.assertTrue(request.call_args.args[0].full_url.endswith('/AngleNaris/ShadowVox/releases/latest'))
        return result

    def test_numeric_version_comparison_and_trusted_release_link(self):
        with patch.object(updates, '__version__', '9.5.0'):
            result = self.check_release({'tag_name': 'v9.10.0', 'html_url': 'https://example.org'})
            self.assertEqual(result['state'], 'newer')
            self.assertEqual(result['url'], 'https://github.com/AngleNaris/ShadowVox/releases/tag/v9.10.0')
            self.assertEqual(self.check_release({'tag_name': 'v9.4.9'})['state'], 'ahead')
        self.assertEqual(self.check_release({'tag_name': 'v'+__version__})['state'], 'current')
        self.assertEqual(self.check_release({'tag_name': 'v99.0.0', 'prerelease': True})['state'], 'error')

    def test_unavailable_rate_limit_and_offline(self):
        cases = [(HTTPError('', 404, '', {}, None), 'unavailable'),
                 (HTTPError('', 403, '', {'X-RateLimit-Remaining': '0'}, None), 'limited'),
                 (URLError('offline'), 'error')]
        for error, expected in cases:
            with self.subTest(expected=expected), patch.object(updates, 'urlopen', side_effect=error):
                self.assertEqual(updates.check_update()['state'], expected)
