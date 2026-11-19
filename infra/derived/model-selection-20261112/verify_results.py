"""Independently check Colab predictions and extend the existing score records."""

import csv
import hashlib
import io
import json
import math
import runpy
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'infra/derived/benchmark-completion-20261112'
IDENTITY = ('path', 'sha256', 'class_id', 'source', 'domain', 'group')
SOURCE_NAMES = {'plantvillage': 'PlantVillage', 'plantdoc': 'PlantDoc', 'plantcity': 'PlantCity', 'ghana_ccmt': 'Ghana CCMT', 'imtiaz': 'Imtiaz', 'openimages_hands': 'Open Images tay', 'plantdoc_other_leaves': 'PlantDoc cây khác', 'rachin': 'Rachin', 'soilclass': 'SoilClass', 'taiwan_tomato': 'Taiwan Tomato', 'tom2024': 'TOM2024'}
verify_metric = runpy.run_path(str(PRIOR / 'verify_and_merge.py'))['verify_metric']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def check_classes(rows, metric):
    verify_metric(rows, metric)
    actual = Counter(row['class_id'] for row in rows)
    predicted = Counter(row['prediction'] for row in rows)
    correct = Counter(row['class_id'] for row in rows if row['class_id'] == row['prediction'])
    assert set(actual) <= set(metric['per_class'])
    for label, value in metric['per_class'].items():
        assert value['support'] == actual[label]
        for key, expected in {
            'precision': correct[label] / predicted[label] if predicted[label] else None,
            'recall': correct[label] / actual[label] if actual[label] else None,
            'f1': 2 * correct[label] / (actual[label] + predicted[label]) if actual[label] else None,
        }.items():
            assert value[key] is None if expected is None else math.isclose(value[key], expected, abs_tol=1e-12), (label, key)


def main():
    result_root = OUT / 'colab-results'
    execution = json.loads((result_root / 'execution.json').read_text())
    receipt = json.loads((OUT / 'input-receipt-v2.json').read_text())
    assert execution['exit_code'] == 0
    assert execution['input_archive_sha256'] == receipt['sha256']
    assert digest(OUT / receipt['file']) == receipt['sha256']
    with ZipFile(OUT / receipt['file']) as archive:
        assert (result_root / 'registration.json').read_bytes() == archive.read('registration.json')
        assert (result_root / 'score.py').read_bytes() == archive.read('score.py')
        registration = json.loads(archive.read('registration.json'))
        datasets = {name: list(csv.DictReader(io.StringIO(archive.read(f'data/{name}.csv').decode('utf-8')))) for name in registration['datasets']}
    summary = json.loads((result_root / 'results/results.json').read_text())
    assert summary['device'] == 'cuda' and summary['input_hashes_unchanged']
    assert summary['registration_sha256'] == digest(result_root / 'registration.json')
    assert summary['versions']['transformers'] == '4.57.6'
    assert summary['versions']['ultralytics'] == '8.4.204'
    expected_runs = {(item['id'], name) for item in registration['models'] for name in item['datasets']}
    assert len(summary['results']) == len(expected_runs) == 26
    assert {(item['model_id'], item['dataset']) for item in summary['results']} == expected_runs
    existing = json.loads((PRIOR / 'metrics-complete.json').read_text(encoding='utf-8'))
    score_index = {tuple(row[:4]): row for row in existing['scores']}
    new_scores, new_classes, checked = [], [], []
    dataset_names = {'taiwan': 'Taiwan/TOM2024', 'historic': 'Test lịch sử', 'validation': 'DINO validation chung'}
    for item in summary['results']:
        metric_file = result_root / 'results' / f"{item['model_id']}-{item['dataset']}-metrics.json"
        assert json.loads(metric_file.read_text()) == item
        file = result_root / 'results' / item['prediction_file']
        assert digest(file) == item['prediction_sha256']
        rows = read_csv(file)
        expected = datasets[item['dataset']]
        assert len(rows) == len(expected)
        assert len({row['path'] for row in rows}) == len(rows)
        assert all(all(a[key] == b[key] for key in IDENTITY) for a, b in zip(rows, expected, strict=True))
        assert all(math.isfinite(float(row['confidence'])) and 0 <= float(row['confidence']) <= 1 for row in rows)
        report = item['metrics']
        slices = [('Toàn bộ', rows, report['all']), ('Lá cà chua thực địa', [r for r in rows if r['domain'] == 'field' and r['class_id'] != 'not_tomato_leaf'], report['field_tomato']), ('Ảnh âm', [r for r in rows if r['class_id'] == 'not_tomato_leaf'], report['not_tomato'])]
        slices += [('Nguồn ' + SOURCE_NAMES.get(source, source), [r for r in rows if r['source'] == source], metric) for source, metric in report['by_source'].items()]
        for cut, members, metric in slices:
            check_classes(members, metric)
            key = (item['label'], dataset_names[item['dataset']], cut, item['epoch'])
            if key in score_index:
                old = score_index[key]
                assert old[4] == len(members)
                assert math.isclose(old[6], metric['accuracy'], abs_tol=1e-12)
                assert math.isclose(old[7], metric['macro_f1'], abs_tol=1e-12)
                continue
            correct = sum(r['class_id'] == r['prediction'] for r in members)
            source = str((result_root / 'results' / f"{item['model_id']}-{item['dataset']}-metrics.json").relative_to(ROOT)).replace('\\', '/')
            note = 'Colab T4. Trọng số cố định. Top-1 trước ngưỡng. Macro-F1 trên lớp có mẫu.'
            if item['dataset'] == 'validation':
                note += ' Tập validation đã dùng trong phát triển.'
            new_scores.append([*key, len(members), correct, metric['accuracy'], metric['macro_f1'], None, sum(v['support'] > 0 for v in metric['per_class'].values()), None, None, None, None, None, None, None, note, source])
            for label, value in metric['per_class'].items():
                new_classes.append([*key, label, value['support'], round(value['recall'] * value['support']) if value['recall'] is not None else 0, value['precision'], value['recall'], value['f1'], None, None, source])
        if item['model_id'] in ('E06', 'E08') and item['dataset'] == 'taiwan':
            assert item['parity_mismatches'] == 0
        checked.append({'id': item['model_id'], 'dataset': item['dataset'], 'images': len(rows), 'prediction_sha256': digest(file), 'macro_f1': report['all']['macro_f1']})
    assert len({tuple(r[:4]) for r in existing['scores'] + new_scores}) == len(existing['scores']) + len(new_scores)
    result = {'new_scores': new_scores, 'new_classes': new_classes, 'checked_runs': checked, 'input_archive_sha256': receipt['sha256'], 'prior_metrics_sha256': digest(PRIOR / 'metrics-complete.json'), 'registration': registration['selection']}
    (OUT / 'verified-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'runs_verified': len(checked), 'images_verified': sum(r['images'] for r in checked), 'new_score_rows': len(new_scores), 'new_class_rows': len(new_classes)}))


if __name__ == '__main__':
    main()
