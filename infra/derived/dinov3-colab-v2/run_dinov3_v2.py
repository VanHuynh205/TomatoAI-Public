"""Continue the sealed DINOv3 experiment on Colab, then score frozen exports."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

CONTENT = Path('/content')
BASE = CONTENT / 'tomato-dinov3-colab-v1'
ROOT = CONTENT / 'tomato-dinov3-colab-v2'
RUN = ROOT / 'run'
MODEL = RUN / 'model'
VERSION = 'dinov3-vits16-field-224-v2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')


def pack(phase, files):
    output = CONTENT / f'dinov3-v2-{phase}-results.zip'
    with ZipFile(output, 'w', ZIP_STORED) as bundle:
        for path, name in files:
            bundle.write(path, name)
    print('RESULT_SHA256', output.name, sha(output), output.stat().st_size, flush=True)


def verify_export(directory, receipt):
    for name, expected in receipt['sha256'].items():
        verify(directory / name, expected)


def train(torch, field):
    receipt = read(ROOT / 'addition-receipt.json')
    initial = read(BASE / 'run/frozen-model.json')
    verify_export(BASE / 'run/model', initial)
    if initial != read(ROOT / 'v1-frozen-model.json'):
        raise ValueError('Unexpected initialization checkpoint')
    verify(BASE / 'holdout-lock.json', receipt['holdout_lock_sha256'])
    verify(ROOT / 'train-val.csv', receipt['train_val_sha256'])
    verify(ROOT / 'dinov3-additions.zip', receipt['addition_zip_sha256'])
    shutil.copytree(BASE / 'data', ROOT / 'data', dirs_exist_ok=True)
    with ZipFile(ROOT / 'dinov3-additions.zip') as additions:
        additions.extractall(ROOT)
    shutil.copyfile(ROOT / 'train-val.csv', ROOT / 'data/train-val.csv')
    rows = field.load_manifest(ROOT / 'data', 'train-val.csv')
    field.validate_splits(rows)
    if len(rows) != sum(receipt['split_count'].values()):
        raise ValueError('Changed manifest count')
    for row in rows:
        verify(ROOT / 'data' / row['path'], row['sha256'])
    params = read(ROOT / 'training-plan.json')
    print('VERIFIED_V2', len(rows), torch.cuda.get_device_name(), params, flush=True)
    field.train(argparse.Namespace(data=ROOT / 'data', output=RUN,
                                  init_export=BASE / 'run/model', **params))
    history = read(RUN / 'history.json')
    if [item['epoch'] for item in history] != list(range(params['epochs'] + 1)):
        raise ValueError('Incomplete continuation history')
    selected = max(history, key=lambda item: item['field']['macro_f1'])
    metadata = read(MODEL / 'leaf-classifier.json')
    metadata.update({
        'model_version': VERSION,
        'preprocessing': read(BASE / 'run/model/leaf-classifier.json')['preprocessing'],
        'test_time_horizontal_flip': False,
    })
    write(MODEL / 'leaf-classifier.json', metadata)
    shutil.copyfile(BASE / 'run/model/LICENSE.md', MODEL / 'LICENSE.md')
    frozen = {
        'status': 'frozen_before_holdout_prediction', 'model_version': VERSION,
        'selected_epoch': selected['epoch'], 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(),
        'sha256': {path.name: sha(path) for path in sorted(MODEL.iterdir())},
        'holdout_lock_sha256': receipt['holdout_lock_sha256'],
        'pilot_manifest_sha256': receipt['pilot']['manifest_sha256'],
        'train_val_sha256': receipt['train_val_sha256'],
        'trainer_sha256': sha(ROOT / 'ai/dinov3_field.py'),
        'runner_sha256': sha(ROOT / 'run_dinov3_v2.py'),
        'training_plan_sha256': sha(ROOT / 'training-plan.json'),
        'initial_export_sha256': initial['sha256'],
    }
    write(RUN / 'frozen-model.json', frozen)
    pack('train', [*((path, f'model/{path.name}') for path in sorted(MODEL.iterdir())),
                   (RUN / 'frozen-model.json', 'frozen-model.json'),
                   (RUN / 'history.json', 'history.json'),
                   (ROOT / 'addition-receipt.json', 'addition-receipt.json'),
                   (ROOT / 'training-plan.json', 'training-plan.json')])


def score(torch, field, directory, root, name):
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoModel

    backbone, loading = AutoModel.from_pretrained(
        directory, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        attn_implementation='sdpa', output_loading_info=True,
    )
    if any(loading.values()):
        raise ValueError('Model did not reload exactly')
    model = field.LeafClassifier(backbone).cuda()
    model.head.load_state_dict(load_file(directory / 'classifier.safetensors'))
    metadata = read(directory / 'leaf-classifier.json')
    prep = metadata['preprocessing']
    rows = field.load_manifest(root, 'test.csv')
    if {row['split'] for row in rows} != {'test'}:
        raise ValueError('Expected test-only rows')
    for row in rows:
        verify(root / row['path'], row['sha256'])
    transform = field.leaf_transform(metadata['size'], prep['mean'], prep['std'], False)
    loader = DataLoader(field.LeafImages(root, rows, transform), batch_size=32, num_workers=2)
    labels, confidence = field.predict(model, loader, torch.device('cuda'))
    target = RUN / f'{name}-predictions.csv'
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ('path', 'sha256', 'class_id', 'source', 'domain',
                                        'group', 'prediction', 'confidence', 'status'))
        writer.writeheader()
        for row, label, value in zip(rows, labels, confidence, strict=True):
            if not 0 <= value <= 1:
                raise ValueError('Non-finite confidence')
            writer.writerow({**{key: row[key] for key in writer.fieldnames[:6]},
                             'prediction': label, 'confidence': value,
                             'status': 'ok' if label != 'not_tomato_leaf' and value >= 0.5 else 'uncertain'})
    print('SCORED', name, len(rows), flush=True)
    del model, backbone
    torch.cuda.empty_cache()
    return target


def test(torch, field):
    receipt = read(ROOT / 'addition-receipt.json')
    frozen = read(RUN / 'frozen-model.json')
    verify_export(MODEL, frozen)
    verify_export(BASE / 'run/model', read(ROOT / 'v1-frozen-model.json'))
    verify(BASE / 'holdout-lock.json', frozen['holdout_lock_sha256'])
    lock = read(BASE / 'holdout-lock.json')
    holdout = ROOT / 'holdout'
    verify(CONTENT / 'dinov3-colab-holdout.zip', '48f2da68ac520097c21ab4064fc9700af4b19ab974c331138fc6c8ccf17254dd')
    with ZipFile(CONTENT / 'dinov3-colab-holdout.zip') as bundle:
        bundle.extractall(holdout)
    pilot_zip = ROOT / 'dinov3-slif-pilot.zip'
    verify(pilot_zip, receipt['pilot']['zip_sha256'])
    with ZipFile(pilot_zip) as bundle:
        bundle.extractall(holdout)
    lock['manifests']['slif'] = {'sha256': frozen['pilot_manifest_sha256'], 'rows': receipt['pilot']['rows']}
    files = []
    for name, expected in lock['manifests'].items():
        verify(holdout / name / 'test.csv', expected['sha256'])
        path = score(torch, field, MODEL, holdout / name, f'dinov3-v2-{name}')
        files.append((path, path.name))
    path = score(torch, field, BASE / 'run/model', holdout / 'slif', 'dinov3-v1-slif')
    files.append((path, path.name))
    pack('test', [*files, (RUN / 'frozen-model.json', 'frozen-model.json')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'test'))
    args = parser.parse_args()
    if not CONTENT.is_dir():
        raise RuntimeError('Run on Colab')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT))
    import torch
    import transformers

    from ai import dinov3_field
    if not torch.cuda.is_available() or transformers.__version__ != '4.57.6':
        raise RuntimeError('Expected Colab GPU and Transformers 4.57.6')
    if args.phase == 'train':
        train(torch, dinov3_field)
    else:
        test(torch, dinov3_field)


if __name__ == '__main__':
    main()
