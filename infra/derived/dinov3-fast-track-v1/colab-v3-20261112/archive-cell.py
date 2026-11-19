import csv
import hashlib
import json
import math
import runpy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from safetensors.torch import load_file

workspace = Path('/content/dinov3-sampler-ab')
registration = json.loads((workspace / 'registration.json').read_text())
environment = json.loads((workspace / 'environment.json').read_text())
shared = registration['shared']
updates = registration['planned_common_updates']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def report_slice(row):
    field = row['field']
    classes = field['per_class']
    details = row['field_diagnostics']
    return {
        'epoch': row['epoch'],
        'updates': row.get('optimizer_steps', 0),
        'loss': row['loss'],
        'field_n': field['n'],
        'field_macro_f1': field['macro_f1'],
        'field_accuracy': field['accuracy'],
        'all_validation_accuracy': row['all']['accuracy'],
        'healthy_recall': classes['healthy']['recall'],
        'early_blight_recall': classes['early_blight']['recall'],
        'bacterial_spot_precision': classes['bacterial_spot']['precision'],
        'bacterial_spot_recall': classes['bacterial_spot']['recall'],
        'bacterial_spot_group_recall': details['bacterial_spot_group_recall'],
        'accepted': details['accepted'],
        'accepted_correct': details['accepted_correct'],
        'healthy_accepted_as_disease': details['healthy_accepted_as_disease'],
        'field_by_source': {name: values['macro_f1'] for name, values in details['field_by_source'].items()},
    }


def pack(target, entries):
    members = {}
    with ZipFile(target, 'x', ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path, name in entries:
            assert path.is_file() and name not in members, name
            archive.write(path, name)
            members[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        archive.writestr('archive-members.json', json.dumps(members, indent=2))
    return {'name': target.name, 'bytes': target.stat().st_size, 'sha256': digest(target)}


assert digest(workspace / 'dinov3_field.py') == registration['trainer_sha256']
assert digest(workspace / 'data/train-val.csv') == registration['manifest_sha256']
check_artifacts = runpy.run_path(str(workspace / 'dinov3_field.py'))['validate_checkpoint_artifacts']
with (workspace / 'data/train-val.csv').open(newline='', encoding='utf-8') as stream:
    train_rows = [row for row in csv.DictReader(stream) if row['split'] == 'train']
assert len(train_rows) == registration['train_rows']

branches = {}
histories = {}
configs = {}
branch_archives = []
common_files = ['registration.json', 'environment.json', 'source-pins.json', 'dinov3_field.py', 'setup-pip.log']
for branch, sampler in registration['samplers'].items():
    directory = workspace / f'sampler-{sampler}'
    history_path = directory / 'history.json'
    exit_file = workspace / f'sampler-{sampler}-exit.json'
    receipt = json.loads(exit_file.read_text())
    assert receipt['exit_code'] == 0 and receipt['optimizer_updates'] == updates[-1]
    assert digest(history_path) == receipt['history_sha256']
    history = json.loads(history_path.read_text())
    histories[branch] = history
    assert [row['optimizer_steps'] for row in history[1:]] == updates
    assert all(math.isfinite(row['loss']) for row in history[1:])
    state = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
    config = state['run']
    assert state['epoch'] == shared['epochs'] - 1 and state['optimizer_steps'] == updates[-1]
    assert state['optimizer']['state'] and state['scaler'].get('scale', 0) > 0
    assert state['history'] == history
    assert config['sampler_version'] == sampler
    assert config['manifest_sha256'] == registration['manifest_sha256']
    assert config['trainer_sha256'] == registration['trainer_sha256']
    assert config['init_export_sha256'] == {
        key: value for key, value in registration['initialization_sha256'].items() if key != 'LICENSE.md'
    }
    configs[branch] = {key: value for key, value in config.items() if key != 'sampler_version'}
    check_artifacts(directory, state)
    for epoch in shared['checkpoint_epochs']:
        with (directory / f'draws-epoch-{epoch}.csv').open(newline='', encoding='utf-8') as stream:
            count = 0
            for index, row in enumerate(csv.DictReader(stream)):
                sampled = train_rows[int(row['train_index'])]
                step = (epoch - 1) * (shared['samples_per_epoch'] // shared['batch']) + index // shared['batch'] + 1
                assert int(row['draw']) == index and row['updated'] == 'True' and int(row['step']) == step
                assert all(row[key] == sampled[key] for key in ('path', 'sha256', 'class_id', 'source', 'group'))
                count += 1
            assert count == shared['samples_per_epoch']

    export = directory / 'model'
    metadata = json.loads((export / 'leaf-classifier.json').read_text())
    best_epoch = state['best_epoch']
    assert metadata['selected_epoch'] == best_epoch
    best = torch.load(directory / f'best-epoch-{best_epoch}.pt', map_location='cpu', weights_only=True)['model']
    serialized = {
        prefix + name: tensor
        for prefix, file in [('backbone.', 'model.safetensors'), ('head.', 'classifier.safetensors')]
        for name, tensor in load_file(export / file, device='cpu').items()
    }
    assert serialized.keys() == best.keys()
    assert all(torch.equal(tensor, best[name]) for name, tensor in serialized.items())
    assert digest(export / 'LICENSE.md') == registration['initialization_sha256']['LICENSE.md']
    del serialized, best, state
    branches[branch] = {
        'sampler': sampler, 'selected_epoch': best_epoch,
        'optimizer_updates': updates[-1], 'draws': shared['epochs'] * shared['samples_per_epoch'],
        'resume_checkpoint_verified': True, 'export_tensors_match_selected_checkpoint': True,
        'started_utc': receipt['started_utc'], 'finished_utc': receipt['finished_utc'],
        'metrics': [report_slice(row) for row in history],
    }
    files = [(path, path.relative_to(workspace).as_posix())
             for path in sorted(directory.rglob('*')) if path.is_file()]
    files += [(workspace / name, name) for name in common_files]
    files += [(exit_file, exit_file.name), (workspace / f'sampler-{sampler}.log', f'sampler-{sampler}.log')]
    packed = pack(Path(f'/content/dinov3-v3-{branch}-sampler-{sampler}.zip'), files)
    branch_archives.append(packed)
    print('V3_BRANCH_ARCHIVED', json.dumps(packed), flush=True)

assert configs['A'] == configs['B']
assert histories['A'][0] == histories['B'][0]
comparison = {
    'scope': 'single_seed_diagnostic_on_historical_validation',
    'validation_rows': registration['validation_rows'],
    'seed': shared['seed'],
    'environment': environment,
    'shared_run_configuration': configs['A'],
    'branches': branches,
    'branch_archives': branch_archives,
}
write_json(workspace / 'comparison.json', comparison)
receipts = [(workspace / name, name) for name in ['comparison.json', *common_files]]
for sampler in registration['samplers'].values():
    path = f'sampler-{sampler}/history.json'
    receipts += [(workspace / path, path),
                 (workspace / f'sampler-{sampler}-exit.json', f'sampler-{sampler}-exit.json'),
                 (workspace / f'sampler-{sampler}.log', f'sampler-{sampler}.log')]
summary_archive = pack(Path('/content/dinov3-v3-receipts.zip'), receipts)
print('V3_BACKUP_READY', json.dumps([summary_archive, *branch_archives]), flush=True)
