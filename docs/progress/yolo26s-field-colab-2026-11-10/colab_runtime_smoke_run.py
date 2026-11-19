# Run after colab_runtime_smoke.py in the same Colab kernel.
# ruff: noqa: F821
review = {"reviewer": "Codex", "scope": "10 deterministic augmented training examples, five source disease classes; runtime smoke only", "decision": "geometry and visible disease features sufficient for runtime smoke", "limitations": ["Not an expert disease relabelling", "Not a full augmentation distribution audit", "Source quality and new domains still need final training review"], "images_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DRIVE.glob("augmentation-*.png"))}}
(DRIVE / "augmentation-review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
run_training()
