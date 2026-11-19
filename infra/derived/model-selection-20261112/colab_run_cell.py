"""Execute frozen checkpoint inference in a fresh Colab directory."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


archive = Path('/content/model-selection-inputs-v2.zip')
expected_hash = '326bf5c4990da632c035d4433452eba926ff4dcc44d5a20b6ba0b349357f422c'
assert archive.stat().st_size == 778357488 and digest(archive) == expected_hash
root = Path('/content/tomato-model-selection-20261112')
root.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as source:
    entries = source.infolist()
    assert len(entries) == len({item.filename for item in entries}) < 10000
    assert sum(item.file_size for item in entries) < 6 * 2**30
    for entry in entries:
        assert (root / entry.filename).resolve().is_relative_to(root.resolve())
        assert not stat.S_ISLNK(entry.external_attr >> 16)
    source.extractall(root)
registration = json.loads((root / 'registration.json').read_text())
assert len(registration['models']) == 18
assert sum(len(item['datasets']) for item in registration['models']) == 26
for name, expected in registration['files'].items():
    assert digest(root / name) == expected, name
print('SELECTION_INPUTS_VERIFIED', expected_hash, '18 checkpoints / 26 runs', flush=True)

runtime = root / 'runtime'
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(runtime)], check=True)
python = runtime / 'bin/python'
with (root / 'setup.log').open('x') as log:
    setup = subprocess.run([str(python), '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--no-input', 'transformers==4.57.6', 'ultralytics==8.4.204'], stdout=log, stderr=subprocess.STDOUT)
if setup.returncode:
    print((root / 'setup.log').read_text()[-4000:], flush=True)
    setup.check_returncode()
env = os.environ.copy()
env.update(PYTHONPATH=str(root), PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', USE_TF='0', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(root / 'yolo-config'))
probe = "import torch, importlib.metadata as m; from transformers import AutoModel; from ai.models import build_model; assert torch.cuda.is_available(); assert m.version('transformers')=='4.57.6'; assert m.version('ultralytics')=='8.4.204'; model=build_model('yolo26s_cls',8); print('SELECTION_RUNTIME_READY',torch.cuda.get_device_name(0),torch.__version__,flush=True)"
subprocess.run([str(python), '-c', probe], cwd=root, env=env, check=True)
started = datetime.now(timezone.utc).isoformat()
with (root / 'run.log').open('x') as log:
    with subprocess.Popen([str(python), '-u', str(root / 'score.py')], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        returncode = process.wait()
execution = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': returncode, 'input_archive_sha256': expected_hash, 'registration_sha256': digest(root / 'registration.json')}
(root / 'execution.json').write_text(json.dumps(execution, indent=2))
result_zip = Path('/content/tomato-model-selection-results-20261112.zip')
with zipfile.ZipFile(result_zip, 'x', compression=zipfile.ZIP_DEFLATED) as result:
    for file in sorted((root / 'results').glob('*')):
        assert file.is_file() and not file.is_symlink()
        result.write(file, 'results/' + file.name)
    for name in ['registration.json', 'score.py', 'setup.log', 'run.log', 'execution.json']:
        result.write(root / name, name)
receipt = {'file': result_zip.name, 'bytes': result_zip.stat().st_size, 'sha256': digest(result_zip), **execution}
print('SELECTION_RESULT_ARCHIVE', json.dumps(receipt), flush=True)
assert returncode == 0, 'Evaluation failed; preserve the result archive and inspect run.log'
