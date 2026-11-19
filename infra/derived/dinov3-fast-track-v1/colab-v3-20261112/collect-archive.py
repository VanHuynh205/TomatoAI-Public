import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

stage = Path(__file__).resolve().parent
branch = sys.argv[1]
sampler = {'A': 'v1', 'B': 'v2'}[branch]
archive_name = f'dinov3-v3-{branch}-sampler-{sampler}.zip'
receipts = json.loads((stage / 'colab-archive-receipts.json').read_text(encoding='utf-8'))
receipt = next(item for item in receipts if item['name'] == archive_name)
download = Path('C:/Users/admin/Downloads') / archive_name


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert download.stat().st_size == receipt['bytes']
assert digest(download) == receipt['sha256']
prefix = f'sampler-{sampler}'
model_files = [
    'model.safetensors', 'classifier.safetensors', 'config.json',
    'preprocessor_config.json', 'leaf-classifier.json', 'LICENSE.md',
]
with ZipFile(download) as archive:
    member_pins = json.loads(archive.read('archive-members.json'))
    names = archive.namelist()
    assert len(names) == len(member_pins) + 1 and set(names) == {*member_pins, 'archive-members.json'}
    expected = {f'{prefix}/last.pt', f'{prefix}/history.json'}
    expected.update(f'{prefix}/step-{step}.pt' for step in (128, 256, 384, 512, 640))
    expected.update(f'{prefix}/draws-epoch-{epoch}.csv' for epoch in range(1, 6))
    expected.update(f'{prefix}/model/{name}' for name in model_files)
    assert expected <= member_pins.keys(), sorted(expected - member_pins.keys())
    for name, pin in member_pins.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in name, name
        assert archive.getinfo(name).file_size == pin['bytes'] > 0, name
        with archive.open(name) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == pin['sha256'], name

    target = stage / archive_name
    with download.open('rb') as source, target.open('xb') as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    assert target.stat().st_size == receipt['bytes'] and digest(target) == receipt['sha256']
    extracted = [f'{prefix}/history.json', *(f'{prefix}/model/{name}' for name in model_files)]
    (stage / prefix / 'model').mkdir(parents=True, exist_ok=False)
    for name in extracted:
        with archive.open(name) as source, (stage / name).open('xb') as output:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        assert digest(stage / name) == member_pins[name]['sha256'], name

verification = {
    'branch': branch,
    'archive': receipt,
    'verified_members': len(member_pins),
    'fixed_updates': [128, 256, 384, 512, 640],
    'model_files': {name: member_pins[name] for name in extracted},
}
with (stage / f'archive-{branch}-verification.json').open('x', encoding='utf-8') as stream:
    json.dump(verification, stream, indent=2)
print(json.dumps({'branch': branch, 'verified_members': len(member_pins), 'bytes': receipt['bytes']}))
