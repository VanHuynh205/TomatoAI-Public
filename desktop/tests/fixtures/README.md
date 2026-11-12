# Browser fixtures

`healthy.png` and `healthy.jpg` are solid green 320 × 240 images generated with Pillow. `uncertain.png` is solid blue. `fake.png` contains plain text. These images have no disease labels or clinical value and do not come from train, validation or TEST.

`care.json` contains synthetic cards for the browser harness only. The healthy card matches `frontend/tests/fixtures/expected.json`; the second synthetic ID tests catalog coverage. No source permission or human review is implied by fixture metadata. Production release validation must reject `artifact_kind: fixture`.
