"""Read-only published-data audits; never discover splits by scanning images."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from csgo_seen10.dataset import CsgoSeen10Dataset, _load_protocol, collate_seen10
from csgo_seen10.prompts import PROMPT_SHA256, PROMPT_VERSION

SPLIT_COUNTS = {"seen_train": 50000, "seen_validation": 5000, "seen_discrete_test": 20000, "seen_continuous": 12800}
SPLIT_FILES = ("train.json", "validation.json", "discrete_test.json", "continuous_clips.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_benchmark(data_root, shared_eval_dir):
    root = Path(data_root).resolve()
    protocol = _load_protocol(shared_eval_dir)
    benchmark = protocol.BenchmarkData(root)
    counts, identities, all_ids = {}, {}, {}
    for split, expected in SPLIT_COUNTS.items():
        rows = benchmark.rows(split)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"{split}: expected {expected} unique published identities, got {len(ids)}/{len(set(ids))}")
        by_map = Counter(row["map_name"] for row in rows)
        if set(by_map) != set(protocol.SEEN_MAPS) or set(by_map.values()) != {expected // 10}:
            raise ValueError(f"{split}: unexpected equal-map population: {dict(by_map)}")
        counts[split] = dict(total=len(ids), by_map=dict(by_map))
        identities[split] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        all_ids[split] = set(ids)
    # Continuous and discrete may be selected under their own released protocol;
    # do not invent a cross-test exclusion rule, but enforce train/val isolation.
    for other in ("seen_validation", "seen_discrete_test", "seen_continuous"):
        if all_ids["seen_train"] & all_ids[other]:
            raise ValueError(f"Published train identities overlap {other}")
    if all_ids["seen_validation"] & (all_ids["seen_discrete_test"] | all_ids["seen_continuous"]):
        raise ValueError("Published validation identities overlap tests")
    clips = benchmark.clips()
    if len(clips) != 200 or any(len(clip["rows"]) != 64 for clip in clips):
        raise ValueError("Expected 200 complete published clips of 64 frames")
    calibration_rel = benchmark.manifest.get("calibration", {}).get("file", "calibration/z_calibration.json")
    metadata_files = ["benchmark_manifest.json", "minimal_dataset_report.json", calibration_rel, "selected_images.sha256"]
    metadata_files.extend(f"splits/seen/{name}/{filename}" for name in protocol.SEEN_MAPS for filename in SPLIT_FILES)
    files = {name: file_sha256(root / name) for name in metadata_files}
    expected_calibration_hash = benchmark.manifest.get("calibration", {}).get("sha256")
    if expected_calibration_hash and files[calibration_rel] != expected_calibration_hash:
        raise ValueError("Calibration file SHA256 differs from published manifest")
    identity = {"files": files, "sample_ids_sha256": identities}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"benchmark_id": "csgo_benchmark_v2", "maps": list(protocol.SEEN_MAPS), "counts": counts,
            "clips": 200, "frames_per_clip": 64, "z_ranges": benchmark.z_ranges,
            "identity": identity, "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256}


def check_target_isolation(data_root, shared_eval_dir):
    """Exercise both test splits while rejecting any attempt to decode a GT image."""
    from PIL import Image
    from unittest.mock import patch
    original_open = Image.open
    opened = []
    radar_root = (Path(data_root) / "radars").resolve()

    def guarded_open(path, *args, **kwargs):
        candidate = Path(path).resolve()
        if radar_root not in candidate.parents:
            raise AssertionError(f"Target isolation violated: Image.open({candidate})")
        opened.append(str(candidate))
        return original_open(path, *args, **kwargs)

    shapes = {}
    with patch.object(Image, "open", guarded_open):
        for split in ("seen_discrete_test", "seen_continuous"):
            dataset = CsgoSeen10Dataset(data_root=str(data_root), shared_eval_dir=str(shared_eval_dir), split=split,
                                       include_target=False, max_samples=2, radar_size=224, target_size=448, pose_mode="text")
            batch = collate_seen10([dataset[index] for index in range(len(dataset))])["data"]["cam2image"]
            if "pixel_values" in batch or "pose_values" in batch:
                raise AssertionError("Aligned inference contains a target or numerical pose tensor")
            shapes[split] = list(batch["cam_values"][0][0].shape)
    return {"target_read": False, "numeric_pose_input": False, "radar_shapes": shapes, "radar_decodes": len(opened)}
