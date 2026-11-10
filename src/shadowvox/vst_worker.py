"""Run offline VST rendering on a separate process's main thread."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import audio, config as C


def serve(folder, render):
    try:
        settings = json.loads((folder/'request.json').read_text(encoding='utf-8'))
        result, stats = render(np.load(folder/'input.npy'), **settings)
        np.save(folder/'output.npy', result)
        (folder/'report.json').write_text(json.dumps(stats), encoding='utf-8')
    except Exception as error:
        (folder/'error.txt').write_text(str(error), encoding='utf-8')
        return 1
    return 0


def render_isolated(flag, mixed, *, checkpoint=None, **settings):
    command = [sys.executable]
    if not getattr(sys, 'frozen', False):
        command += ['-B', str(C.ROOT/'desktop.py')]
    env = os.environ.copy()
    env.pop('PYINSTALLER_RESET_ENVIRONMENT', None)
    with tempfile.TemporaryDirectory(prefix='shadowvox-vst-') as directory:
        folder = Path(directory)
        np.save(folder/'input.npy', mixed)
        (folder/'request.json').write_text(json.dumps(settings), encoding='utf-8')
        if checkpoint is not None:
            checkpoint()
        with subprocess.Popen(command+[flag, directory], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, **audio.subprocess_options()) as process:
            try:
                while True:
                    try:
                        _, stderr = process.communicate(timeout=.1 if checkpoint is not None else None)
                        break
                    except subprocess.TimeoutExpired:
                        checkpoint()
            except InterruptedError:
                process.kill()
                raise
        if process.returncode:
            error = folder/'error.txt'
            detail = error.read_text(encoding='utf-8') if error.is_file() else stderr.decode(errors='replace')
            raise RuntimeError(f'ZL 母带处理失败：{detail or process.returncode}')
        return np.load(folder/'output.npy'), json.loads((folder/'report.json').read_text(encoding='utf-8'))
