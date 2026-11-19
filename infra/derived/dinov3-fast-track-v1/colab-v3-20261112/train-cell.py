import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

archive = Path('/content/dinov3-sampler-ab.zip')
workspace = Path('/content/dinov3-sampler-ab')
package_sha256 = '4e986d5d7fc2061a8ed98eb2a3ddc08916d09d2db2ce0b41709f866b6b39b8d9'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


assert archive.stat().st_size == 237736260 and digest(archive) == package_sha256
workspace.mkdir(exist_ok=False)
with ZipFile(archive) as source:
    source.extractall(workspace)

registration_path = workspace / 'registration.json'
registration = json.loads(registration_path.read_text(encoding='utf-8'))
assert digest(registration_path) == 'ef2ae999cd31ed18d6526dbbf7d5f068202a8877c5ebbfb260e2f2b1e261c502'
trainer = workspace / 'dinov3_field.py'
assert digest(trainer) == registration['trainer_sha256']
assert digest(workspace / 'data/train-val.csv') == registration['manifest_sha256']
initialization = workspace / 'v1-model'
for name, expected in registration['initialization_sha256'].items():
    assert digest(initialization / name) == expected, name
print('V3_INPUTS_VERIFIED', package_sha256, flush=True)

deps = workspace / 'runtime-deps'
pip_log = workspace / 'setup-pip.log'
print('V3_DEPENDENCIES installing Transformers 4.57.6 in the run directory', flush=True)
with pip_log.open('x', encoding='utf-8') as output:
    setup = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
         '--no-input', '--target', str(deps), 'transformers==4.57.6'],
        stdout=output, stderr=subprocess.STDOUT,
    )
if setup.returncode:
    print(pip_log.read_text(encoding='utf-8', errors='replace')[-3000:], flush=True)
    setup.check_returncode()

env = os.environ.copy()
env['PYTHONPATH'] = str(deps) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
env['HF_HUB_OFFLINE'] = '1'
env['TRANSFORMERS_OFFLINE'] = '1'
env['USE_TF'] = '0'
probe = """
import importlib.metadata as meta
import json
import sys
import torch
import torchvision
from transformers import AutoModel, AutoImageProcessor
assert torch.cuda.is_available()
assert meta.version('transformers') == '4.57.6'
print(json.dumps({
    'python': sys.version.split()[0],
    'gpu': torch.cuda.get_device_name(0),
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'cuda': torch.version.cuda,
    'packages': {name: meta.version(name) for name in (
        'torch', 'torchvision', 'transformers', 'huggingface-hub',
        'tokenizers', 'safetensors', 'pillow', 'numpy',
    )},
}, sort_keys=True))
"""
environment = json.loads(subprocess.check_output([sys.executable, '-c', probe], env=env, text=True))
environment.update({
    'package_sha256': package_sha256,
    'registration_sha256': digest(registration_path),
    'checked_utc': utc_now(),
})
(workspace / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
print('V3_ENVIRONMENT', json.dumps(environment), flush=True)

shared = registration['shared']
arguments = [
    sys.executable, '-u', str(trainer),
    '--data', str(workspace / 'data'), '--init-export', str(initialization),
    '--size', str(shared['size']), '--epochs', str(shared['epochs']),
    '--samples', str(shared['samples_per_epoch']), '--batch', str(shared['batch']),
    '--workers', str(shared['workers']), '--seed', str(shared['seed']),
    '--tuning', shared['tuning'], '--audit-draws', '--checkpoint-epochs',
    *map(str, shared['checkpoint_epochs']),
]
for branch, sampler in registration['samplers'].items():
    output_dir = workspace / f'sampler-{sampler}'
    assert not output_dir.exists(), output_dir
    command = arguments + ['--output', str(output_dir), '--sampler', sampler]
    started = utc_now()
    with (workspace / f'sampler-{sampler}.log').open('x', encoding='utf-8') as log:
        with subprocess.Popen(
            command, cwd=workspace, env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as job:
            print('V3_BRANCH_STARTED', branch, sampler, job.pid, started, flush=True)
            for line in job.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            exit_code = job.wait()

    receipt = {
        'branch': branch, 'sampler': sampler, 'command': command,
        'started_utc': started, 'finished_utc': utc_now(), 'exit_code': exit_code,
    }
    if exit_code:
        (workspace / f'sampler-{sampler}-exit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        raise subprocess.CalledProcessError(exit_code, command)

    history_file = output_dir / 'history.json'
    history = json.loads(history_file.read_text(encoding='utf-8'))
    assert [row['epoch'] for row in history] == list(range(shared['epochs'] + 1))
    assert [row['optimizer_steps'] for row in history[1:]] == registration['planned_common_updates']
    assert (output_dir / 'last.pt').is_file()
    for step in registration['planned_common_updates']:
        assert (output_dir / f'step-{step}.pt').is_file(), step
    shutil.copy2(initialization / 'LICENSE.md', output_dir / 'model/LICENSE.md')
    receipt['history_sha256'] = digest(history_file)
    receipt['optimizer_updates'] = history[-1]['optimizer_steps']
    receipt['last_field_macro_f1'] = history[-1]['field']['macro_f1']
    (workspace / f'sampler-{sampler}-exit.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('V3_BRANCH_DONE', json.dumps(receipt), flush=True)

print('V3_AB_TRAINING_DONE', utc_now(), flush=True)
