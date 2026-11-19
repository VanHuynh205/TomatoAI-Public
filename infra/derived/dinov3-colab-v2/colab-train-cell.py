"""Notebook cell; execute only in the existing Colab runtime."""

import hashlib
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from google.colab import files

package = Path('/content/dinov3-v2-package.zip')
assert hashlib.sha256(package.read_bytes()).hexdigest() == 'a76e03fce4e9830148fcd9801ae32746fa51eb97bec460a5dd952b1a7c6cacbe'
v2_root = Path('/content/tomato-dinov3-colab-v2')
assert not v2_root.exists(), 'Inspect an existing v2 run before starting or resuming'
with ZipFile(package) as archive:
    archive.extractall(v2_root)


def run_v2(phase):
    command = [sys.executable, '-u', str(v2_root / 'run_dinov3_v2.py'), phase]
    with (v2_root / f'{phase}.log').open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as job:
            for line in job.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            if job.wait():
                raise subprocess.CalledProcessError(job.returncode, command)


run_v2('train')
files.download('/content/dinov3-v2-train-results.zip')
