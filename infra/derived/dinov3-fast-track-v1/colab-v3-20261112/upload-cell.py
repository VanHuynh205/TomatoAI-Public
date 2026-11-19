from pathlib import Path
import hashlib

v3_archive = Path('/content/dinov3-sampler-ab.zip')
if not v3_archive.exists():
    from google.colab import files
    print('V3_UPLOAD_PICKER', flush=True)
    files.upload(target_dir=str(v3_archive.parent))
assert v3_archive.stat().st_size == 237736260, 'Wait for the 238 MB archive to finish uploading'
with v3_archive.open('rb') as stream:
    v3_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
assert v3_digest == '4e986d5d7fc2061a8ed98eb2a3ddc08916d09d2db2ce0b41709f866b6b39b8d9'
print('V3_UPLOAD_VERIFIED', v3_archive.stat().st_size, v3_digest, flush=True)
