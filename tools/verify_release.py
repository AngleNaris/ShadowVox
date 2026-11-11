"""核对全量包，并用同一真实音频比较源码、独立 EXE 与报告导出。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.utils.win32.versioninfo import read_version_info_from_executable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shadowvox import AUTHOR, __version__, mixing, pipeline
from shadowvox.jobs import DESKTOP_RENDERER
from shadowvox.reporting import build


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(code):
    return code.replace(co_filename='', co_consts=tuple(
        normalize(value) if isinstance(value, types.CodeType) else value for value in code.co_consts))


def check_archive(executable):
    package = CArchiveReader(str(executable))
    modules = package.open_embedded_archive('PYZ.pyz')
    checked = []
    for name in ('__init__', 'align', 'hubert_alignment', 'full_align', 'lyric_anchors', 'pipeline', 'live_preview',
                 'psola', 'signalsmith', 'separate', 'vocal_gate', 'jobs', 'updates',
                 'mixing', 'density', 'mastering', 'vst_worker', 'gui', 'gui_bridge', 'reporting', 'report_analysis', 'denoise'):
        path = ROOT/'src/shadowvox'/f'{name}.py'
        key = 'shadowvox' if name == '__init__' else f'shadowvox.{name}'
        assert normalize(modules.extract(key)) == normalize(compile(path.read_bytes(), str(path), 'exec')), key
        checked.append(key)
    assets = []
    for path in (ROOT/'src/shadowvox/ui').rglob('*'):
        if path.is_file():
            key = str(Path('shadowvox/ui')/path.relative_to(ROOT/'src/shadowvox/ui'))
            assert package.extract(key) == path.read_bytes(), key
            assets.append(key)
    helper = 'bin\\shadowvox-denoise.exe'
    assert hashlib.sha256(package.extract(helper)).hexdigest() == digest(ROOT/'tools/vendor/deepfilternet/shadowvox-denoise.exe')
    helper = 'bin\\shadowvox-signalsmith.exe'
    assert hashlib.sha256(package.extract(helper)).hexdigest() == digest(ROOT/'tools/vendor/signalsmith/shadowvox-signalsmith.exe')
    model_files = []
    for path in (ROOT/'models').rglob('*'):
        key = str(path.relative_to(ROOT))
        if path.is_file() and key in package.toc:
            assert hashlib.sha256(package.extract(key)).hexdigest() == digest(path), key
            model_files.append(key)
    assert 'models\\hubertfa\\model.onnx' in model_files
    for name in ('Compressor', 'Limiter'):
        zl = f'plugins\\ZL {name}.vst3\\Contents\\x86_64-win\\ZL {name}.vst3'
        assert hashlib.sha256(package.extract(zl)).hexdigest() == digest(ROOT/zl)
    assert 'licenses\\zl-compressor\\ZL-AGPL-3.0.txt' in package.toc
    for key in ('bin\\ffmpeg.exe','bin\\shadowvox-signalsmith.exe',
                'licenses\\signalsmith\\stretch\\LICENSE.txt','licenses\\signalsmith\\linear\\LICENSE.txt',
                'models\\whisper-base.pt','models\\fcpe.onnx','models\\Kim_Vocal_2.onnx'):
        assert key in package.toc, key
    info = read_version_info_from_executable(str(executable))
    assert __version__ in str(info), 'Windows 版本属性不一致'
    assert AUTHOR in str(info), 'Windows 作者署名不一致'
    manifest = json.loads(executable.with_suffix('.release.json').read_text(encoding='utf-8'))
    assert manifest['version'] == __version__ and manifest['executable']['sha256'] == digest(executable)
    return dict(modules=checked, assets=assets, model_files=model_files, denoise_helper_matches=True, windows_version=__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True, help='已有的真实音频输入请求 JSON')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, help='复用同一源码已生成的对照报告')
    args = parser.parse_args()
    executable, folder = args.exe.resolve(), args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(version=__version__, archive=check_archive(executable))
    request = json.loads(args.request.read_text(encoding='utf-8'))
    request.setdefault('master_density',1.)
    request.setdefault('fixed_master',True)
    request.setdefault('mute_reference_gaps',True)
    if args.source_report:
        source = json.loads(args.source_report.read_text(encoding='utf-8'))
        source_report = args.source_report.resolve()
    else:
        output = folder/'source/reference.wav'
        output.parent.mkdir(parents=True, exist_ok=True)
        capture = {}
        with (folder/'source/pipeline.log').open('w', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
            source = pipeline.run(request['dry'], request['song'], output, work_dir=request.get('work'),
                opts=pipeline.Options(strength=request.get('strength', .75), pitch_mode='register',
                    max_shift_cents=450, mix_effects=request.get('mix_effects', True), comparison=True,
                    renderer=DESKTOP_RENDERER, rhythm_mode='full', master_density=request['master_density'], fixed_master=request['fixed_master'],
                    reuse_stems=request.get('reuse_stems', True), mute_reference_gaps=request['mute_reference_gaps'],
                    dry_has_backing=request.get('dry_has_backing', False)),
                capture=capture, log=emit)
        source['audio_analysis'] = build(source, source['comparison_before'], output, capture=capture)
        source_report = output.with_suffix('.report.json')
        source_report.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    assert source['app_version'] == __version__
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('QT_QPA_PLATFORM', None)
    env['PATH'] = str(Path(os.environ['SystemRoot'])/'System32')
    env['LOCALAPPDATA'] = str(folder/'userdata')
    env['APPDATA'] = str(folder/'roaming')
    env['SHADOWVOX_OFFLINE_TEST'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['HTTP_PROXY'] = env['HTTPS_PROXY'] = 'http://127.0.0.1:1'
    request['output'] = str(folder/'frozen.wav')
    request['work'] = str(folder/'fresh-cache')
    request['reuse_stems'] = False
    request_path = folder/'process-request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    start = time.monotonic()
    process = subprocess.run([str(executable), '--process', str(request_path)], cwd=folder,
                             env=env, creationflags=subprocess.CREATE_NO_WINDOW, timeout=1800)
    assert process.returncode == 0, (folder/'frozen.log').read_text(encoding='utf-8')
    frozen = json.loads((folder/'frozen.report.json').read_text(encoding='utf-8'))
    assert frozen['app_version'] == __version__
    assert frozen['mixing']['mix_preset'] == mixing.MIX_PRESET
    assert frozen['mixing']['duck_band_hz'] == [250, 4000]
    assert frozen['mixing']['max_broadband_duck_db'] == frozen['mixing']['max_low_mid_duck_db'] == 0.
    assert 0. < frozen['mixing']['max_band_duck_db'] <= 2.125
    if request['master_density']:
        assert frozen['mixing']['density']['applied']
    if request['fixed_master']:
        assert frozen['mixing']['readback_verified'] and frozen['mixing']['measured_output_true_peak_dbtp'] <= -.5
        assert frozen['mixing']['target_reached']
    rhythm = frozen['rhythm']
    assert rhythm['algorithm'] == 'hubertfa-zh+signalsmith-v1'
    assert rhythm['language'] == 'zh' and rhythm['lyric_anchors'] > 0
    assert rhythm['order_preserved'] and rhythm['continuous_source_coverage']
    assert rhythm['rate_min'] >= .9 and rhythm['rate_max'] <= 1.1
    assert frozen['denoise']['model'] == 'DeepFilterNet3'
    actual, sr = sf.read(folder/'frozen.wav', dtype='float32', always_2d=True)
    expected, expected_sr = sf.read(source['output'], dtype='float32', always_2d=True)
    assert sr == expected_sr == 44100 and actual.shape == expected.shape and actual.shape[1] == 2
    assert np.isfinite(actual).all() and len(actual)/sr >= frozen['dry_seconds']-.01
    difference = float(np.max(np.abs(actual-expected)))
    assert difference <= 2e-6, difference
    result['process'] = dict(exit_code=process.returncode, elapsed_seconds=time.monotonic()-start,
        samples=len(actual), sample_rate=sr, source_max_sample_difference=difference,
        mix_preset=frozen['mixing']['mix_preset'], rhythm=rhythm, measurement=source['audio_analysis']['output'])
    gui = folder/'gui'; gui.mkdir(exist_ok=True)
    (gui/'report-request.json').write_text(json.dumps({'report':str(source_report)}), encoding='utf-8')
    (gui/'preview-request.json').write_text(json.dumps(dict(song=request['song'], dry=request['dry'],
        before=source['comparison_before'], after=str(folder/'frozen.wav'))), encoding='utf-8')
    process = subprocess.run([str(executable), '--gui-check', str(gui)], cwd=folder, env=env,
                             creationflags=subprocess.CREATE_NO_WINDOW, timeout=100)
    assert process.returncode == 0
    state = json.loads((gui/'frozen-gui.json').read_text(encoding='utf-8'))
    assert state['app_version'] == __version__ and state['visible'] and state['report_png']
    assert state['sound_values'] == dict(correction=50, beauty=50, fusion=50, ratio=0, mixdown=100)
    assert state['report_summary_rows'] >= 8
    assert (ROOT/'src/shadowvox/ui/report.js').read_text(encoding='utf-8') in (gui/'frozen-report.html').read_text(encoding='utf-8')
    result['gui'] = state
    result['passed'] = True
    (folder/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=True, version=__version__, samples=len(actual),
        source_max_sample_difference=difference, report_png=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
