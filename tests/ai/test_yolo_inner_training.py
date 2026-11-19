from importlib import import_module

import pytest

MODULE = "docs.progress.yolo-image-recovery.inner_training"


def test_inner_split_keeps_whole_groups_and_only_train() -> None:
    rows = [
        {
            "sample_id": f"{source}:{name}{group}{i}",
            "class_id": name,
            "component_id": f"{source}:{name}{group}",
            "split": split,
        }
        for source in ("pv", "plantdoc")
        for name in ("healthy", "bacterial_spot")
        for group, split in enumerate(("train", "train", "train", "val", "test"))
        for i in range(group + 1)
    ]
    method = import_module(MODULE).inner_split
    train, dev = method(rows)
    assert (train, dev) == method(list(reversed(rows)))
    assert set(train).isdisjoint(dev)
    assert set(train) | set(dev) == {
        r["sample_id"] for r in rows if r["split"] == "train"
    }
    assert all(
        len({r["component_id"] for r in rows if r["sample_id"] in subset}) == count
        for subset, count in ((train, 8), (dev, 4))
    )
    assert {r["component_id"] for r in rows if r["sample_id"] in train}.isdisjoint(
        r["component_id"] for r in rows if r["sample_id"] in dev
    )


def test_inner_split_rejects_components_crossing_locked_splits() -> None:
    rows = [
        {
            "sample_id": "pv:1",
            "class_id": "healthy",
            "component_id": "a",
            "split": "train",
        },
        {
            "sample_id": "pv:2",
            "class_id": "healthy",
            "component_id": "b",
            "split": "train",
        },
        {
            "sample_id": "pv:3",
            "class_id": "healthy",
            "component_id": "a",
            "split": "val",
        },
    ]
    with pytest.raises(ValueError, match="locked splits"):
        import_module(MODULE).inner_split(rows)
