from __future__ import annotations

import json

from vos_memory_inspector.lvos import build_lvosv2_evaluation_manifest


def test_lvos_manifest_uses_object_frame_range_and_attributes(tmp_path):
    split = tmp_path / "val"
    split.mkdir()
    (split / "val_meta.json").write_text(
        json.dumps(
            {
                "videos": {
                    "demo": {
                        "objects": {
                            "3": {"frame_range": {"start": 10, "end": 40, "frame_nums": 31}}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (split / "val_meta_attribute.json").write_text(
        json.dumps({"videos": {"demo": {"attributes": ["long-term", "occlusion"]}}}),
        encoding="utf-8",
    )

    manifest = build_lvosv2_evaluation_manifest(
        tmp_path, min_prefix_frames=2, min_future_frames=3
    )

    assert manifest["sequence_count"] == 1
    assert manifest["case_count"] == 3
    assert all(case["object_id"] == "3" for case in manifest["cases"])
    assert all(case["tags"] == ["long-term", "occlusion"] for case in manifest["cases"])
    assert all(case["future_gt_available"] is True for case in manifest["cases"])

