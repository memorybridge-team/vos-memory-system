"""DAVIS-style region J and contour F metrics for indexed VOS masks."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # Slow but exact fallback used by lightweight test installs.
    distance_transform_edt = None


def region_jaccard(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    union = np.logical_or(prediction, ground_truth).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, ground_truth).sum() / union)


def boundary_map(mask):
    mask = np.asarray(mask, dtype=bool)
    east = np.zeros_like(mask)
    south = np.zeros_like(mask)
    southeast = np.zeros_like(mask)
    east[:, :-1] = mask[:, 1:]
    south[:-1, :] = mask[1:, :]
    southeast[:-1, :-1] = mask[1:, 1:]
    boundary = ((mask != east) | (mask != south) | (mask != southeast))
    boundary[-1, :] = mask[-1, :] != east[-1, :]
    boundary[:, -1] = mask[:, -1] != south[:, -1]
    boundary[-1, -1] = False
    return boundary


def boundary_f(prediction, ground_truth, tolerance=0.008):
    pred_boundary = boundary_map(prediction)
    gt_boundary = boundary_map(ground_truth)
    pred_count, gt_count = pred_boundary.sum(), gt_boundary.sum()
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0
    radius = max(1, int(math.ceil(tolerance * math.hypot(*pred_boundary.shape))))
    if distance_transform_edt is not None:
        pred_match = pred_boundary & (distance_transform_edt(~gt_boundary) <= radius)
        gt_match = gt_boundary & (distance_transform_edt(~pred_boundary) <= radius)
    else:
        pred_match = pred_boundary & _dilate(gt_boundary, radius)
        gt_match = gt_boundary & _dilate(pred_boundary, radius)
    precision = pred_match.sum() / pred_count
    recall = gt_match.sum() / gt_count
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def _dilate(mask, radius):
    result = np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            result[target_y, target_x] |= mask[source_y, source_x]
    return result


def evaluate_indexed_masks(video, predictions, prompt_frames):
    """Evaluate labeled, non-prompt frames and respect LVOS object ranges."""
    per_object = []
    object_ids = sorted(int(value) for value in prompt_frames)
    for object_id in object_ids:
        prompt = int(prompt_frames[object_id])
        metadata = video.objects.get(str(object_id), video.objects.get(object_id, {}))
        frame_range = metadata.get("frame_range", {}) if metadata else {}
        start = int(Path(str(frame_range.get("start", prompt))).stem)
        end = int(Path(str(frame_range.get("end", max(video.frame_ids)))).stem)
        if video.dataset_name == "lvos":
            # LVOS v1 evaluates every fifth source frame and excludes the
            # object's first and last annotated frames (official toolkit).
            candidate_frames = list(range(start, end, 5))[1:-1]
        else:
            candidate_frames = [frame for frame in sorted(video.masks)
                                if start <= frame <= end and frame != prompt]
            if video.dataset_name == "davis2017" and candidate_frames:
                # DAVIS semi-supervised evaluation excludes its final frame.
                candidate_frames = candidate_frames[:-1]
        frames = []
        for frame_id in candidate_frames:
            if frame_id == prompt:
                continue
            if frame_id not in video.masks:
                raise ValueError(f"Missing ground truth for evaluation frame {frame_id}")
            gt_path = video.masks[frame_id]
            if frame_id not in predictions:
                raise ValueError(f"Missing prediction for labeled frame {frame_id}")
            gt = np.asarray(Image.open(gt_path)) == object_id
            pred = np.asarray(predictions[frame_id]) == object_id
            j_score = region_jaccard(pred, gt)
            f_score = boundary_f(pred, gt)
            frames.append({"frame_id": frame_id, "j": j_score, "f": f_score,
                           "j_and_f": (j_score + f_score) / 2})
        if not frames:
            continue
        per_object.append({
            "object_id": object_id,
            "evaluation_group": video.evaluation_group,
            "prompt_frame": prompt,
            "frames": frames,
            "j": float(np.mean([item["j"] for item in frames])),
            "f": float(np.mean([item["f"] for item in frames])),
            "j_and_f": float(np.mean([item["j_and_f"] for item in frames])),
        })
    if not per_object:
        raise ValueError(f"No non-prompt labeled frames to score for {video.name}")
    return {
        "objects": per_object,
        "j": float(np.mean([item["j"] for item in per_object])),
        "f": float(np.mean([item["f"] for item in per_object])),
        "j_and_f": float(np.mean([item["j_and_f"] for item in per_object])),
    }
