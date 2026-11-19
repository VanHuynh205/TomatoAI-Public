import importlib.metadata as meta
import json
import shutil
import sys
from pathlib import Path

import torch

versions = {}
for name in ('torch', 'torchvision', 'transformers', 'safetensors', 'pillow', 'numpy'):
    try:
        versions[name] = meta.version(name)
    except meta.PackageNotFoundError:
        versions[name] = None
cuda = torch.cuda.is_available()
print('V3_PREFLIGHT', json.dumps({
    'python': sys.version.split()[0],
    'packages': versions,
    'cuda_available': cuda,
    'gpu': torch.cuda.get_device_name(0) if cuda else None,
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if cuda else None,
    'disk_free_bytes': shutil.disk_usage('/content').free,
    'package_present': Path('/content/dinov3-sampler-ab.zip').exists(),
    'run_directory_present': Path('/content/dinov3-sampler-ab').exists(),
}, sort_keys=True), flush=True)
assert cuda, 'Colab GPU runtime required'
