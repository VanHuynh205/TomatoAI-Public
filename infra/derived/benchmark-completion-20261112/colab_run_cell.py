from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

archive = Path('/content/benchmark-inputs.zip')
assert archive.stat().st_size == 714140491
with archive.open('rb') as stream:
    archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert archive_hash == '6c7978fe80293d52879bb88ddefac54e9b63f67792ce2125d8f933f862770b78'
root = Path('/content/tomato-benchmark-20261112')
assert not root.exists(), 'Use the existing result/log to investigate; do not overwrite a run'
root.mkdir()
with zipfile.ZipFile(archive) as data:
    for name in data.namelist():
        assert (root / name).resolve().is_relative_to(root.resolve())
    data.extractall(root)
print('BENCH_INPUTS_VERIFIED', archive_hash, flush=True)
deps = root / 'dependencies'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', str(deps), 'transformers==4.57.6'], check=True)
env = os.environ.copy()
env['PYTHONPATH'] = str(deps) + os.pathsep + str(root)
env['PYTHONUNBUFFERED'] = '1'
env['HF_HUB_OFFLINE'] = '1'
env['TOKENIZERS_PARALLELISM'] = 'false'
with (root / 'run.log').open('w') as logfile:
    process = subprocess.Popen([sys.executable, '-u', str(root / 'colab_predict.py')], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logfile.write(line)
        logfile.flush()
        print(line, end='', flush=True)
    returncode = process.wait()
assert returncode == 0, f'Benchmark failed; retain run.log. Exit code {returncode}'
result_zip = Path('/content/tomato-benchmark-results-20261112.zip')
with zipfile.ZipFile(result_zip, 'x', compression=zipfile.ZIP_DEFLATED) as result:
    for file in sorted((root / 'results').glob('*')):
        result.write(file, 'results/' + file.name)
    for name in ['inputs.json', 'colab_predict.py', 'run.log']:
        result.write(root / name, name)
with result_zip.open('rb') as stream:
    result_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'name': result_zip.name, 'bytes': result_zip.stat().st_size, 'sha256': result_hash, 'input_archive_sha256': archive_hash}
print('BENCH_RESULT_ARCHIVE', json.dumps(receipt), flush=True)
from google.colab import files
files.download(str(result_zip))
