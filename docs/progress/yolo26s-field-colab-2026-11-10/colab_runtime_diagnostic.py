import inspect
import json
from pathlib import Path

import torch

from ultralytics.engine.trainer import BaseTrainer
checkpoint_path = Path('/content/drive/MyDrive/TomatoAI/yolo26s-field-2026-11-10/runtime-smoke-v2/train/weights/last.pt')
checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(json.dumps({'checkpoint': str(checkpoint_path), 'epoch': checkpoint_state.get('epoch'), 'optimizer_state_count': len(checkpoint_state.get('optimizer', {}).get('state', {})), 'scaler': checkpoint_state.get('scaler'), 'updates': checkpoint_state.get('updates'), 'nbs': checkpoint_state['train_args'].get('nbs'), 'batch': checkpoint_state['train_args'].get('batch')}, indent=2))
for method in (BaseTrainer._setup_train, BaseTrainer._do_train):
    lines = inspect.getsource(method).splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if any(word in line for word in ('accumulate', 'last_opt_step', 'optimizer_step()', 'ni =', 'resume_training')):
            selected.update(range(max(0, i-1), min(len(lines), i+3)))
    print(method.__name__)
    print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected)))
print('CHECKPOINT_DIAGNOSTIC_COMPLETE')
