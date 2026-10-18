"""
enroll_quality.py
=================
Enrollment-specific face acceptance filter (Stage 7+).

Deliberately SEPARATE from quality.py/assess_quality() which is the runtime
recognition gate. Enrollment quality has a completely different purpose:
  - Runtime gate: protect a SINGLE recognition decision from one bad frame
  - Enrollment gate: decide whether a frame contains USABLE IDENTITY INFORMATION
    that should be stored in the gallery

Key differences from the runtime gate:
  - NO angle rejection (angled frames are exactly what we want in the gallery)
  - NO occlusion rejection (masked frames are what periocular needs)
  - Much lower sharpness floor (just blocks pure motion smear)
  - Much wider brightness band (captures dimly-lit walking conditions)
  - Returns pose angles and mask signal so the caller can BUCKET frames,
    not reject them based on angle/occlusion

CPU-only, ~0.3ms per call. No GPU or deep model dependency.
"""

import cv2
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from quality import _estimate_pose, _compute_sharpness, _compute_brightness


# ─────────────────────────────────────────────────────────────────────────────
#  Thresholds — deliberately loose for enrollment
# ─────────────────────────────────────────────────────────────────────────────

# Enrollment-only thresholds (not shared with runtime gate)
ENROLL_MIN_FACE_PX   = 60      # Require a decent size for gallery templates
ENROLL_MIN_SHARPNESS = 5.0     # Just enough to reject pure motion smear
ENROLL_MIN_BRIGHTNESS = 15     # Very dark is still usable identity info
ENROLL_MAX_BRIGHTNESS = 245    # Blown-out is still usable (some detail remains)

# Bucket classification thresholds
ENROLL_FRONTAL_YAW_DEG = 20.0  # Yaw ≤ 20° = frontal bucket, > 20° = angled bucket


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_FRONTAL_UNMASKED  = "frontal_unmasked"
BUCKET_ANGLED_UNMASKED   = "angled_unmasked"
BUCKET_MASKED_FRONTAL    = "masked_frontal"
BUCKET_MASKED_ANGLED     = "masked_angled"

ALL_BUCKETS = [
    BUCKET_FRONTAL_UNMASKED,
    BUCKET_ANGLED_UNMASKED,
    BUCKET_MASKED_FRONTAL,
    BUCKET_MASKED_ANGLED,
]


@dataclass
class EnrollResult:
    """Result of enrollment quality assessment for a single frame."""
    accepted: bool          # True if frame passes the loose floor checks
    reject_reason: str      # "" if accepted, else "TOO_SMALL"/"TOO_BLURRY"/"BAD_LIGHTING"
    sharpness: float        # Laplacian variance — used for within-bucket ranking
    brightness: float       # Mean pixel intensity
    face_size: int          # min(w, h) of detected face bbox
    yaw: float              # Head yaw in degrees (positive = turning right)
    pitch: float            # Head pitch in degrees (positive = looking down)
    is_masked: bool         # True if mask heuristic fired
    bucket: str             # One of ALL_BUCKETS, or "" if rejected


def assess_enroll_quality(
    face_crop_bgr: np.ndarray,
    face_obj=None,
    frontal_yaw_threshold: float = ENROLL_FRONTAL_YAW_DEG,
    frame_w: int = None,
    frame_h: int = None,
) -> EnrollResult:
    """
    Assess whether a face crop is useful for enrollment.

    Applies a very loose quality floor (not angle/occlusion checks),
    then classifies the accepted frame into a condition bucket.

    Args:
        face_crop_bgr: BGR crop of the detected face region.
        face_obj: InsightFace face object (for landmarks + mask check).
                  If None, pose and mask checks are skipped (frame still accepted
                  if it passes the size/sharpness/brightness floor).
        frontal_yaw_threshold: Yaw degrees below which a frame is "frontal".
                               Default 20°. Raise to 25-30° for overhead cameras.
        frame_w: Full original frame width. Required for correct pose estimation
                 when face_obj.kps are in full-frame coordinates (enrollment mode).
                 If None, falls back to face crop dimensions (less accurate).
        frame_h: Full original frame height. Same purpose as frame_w.

    Returns:
        EnrollResult with accept/reject verdict, pose angles, mask flag, and bucket.
    """
    h, w = face_crop_bgr.shape[:2]
    face_size = min(w, h)
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    # ── Floor 1: Minimum face size ───────────────────────────────────────────
    if face_size < ENROLL_MIN_FACE_PX:
        return EnrollResult(
            accepted=False, reject_reason="TOO_SMALL",
            sharpness=0.0, brightness=0.0, face_size=face_size,
            yaw=0.0, pitch=0.0, is_masked=False, bucket="",
        )

    # ── Floor 2: Minimum sharpness (just blocks pure smear) ─────────────────
    sharpness = _compute_sharpness(gray)
    if sharpness < ENROLL_MIN_SHARPNESS:
        return EnrollResult(
            accepted=False, reject_reason="TOO_BLURRY",
            sharpness=sharpness, brightness=0.0, face_size=face_size,
            yaw=0.0, pitch=0.0, is_masked=False, bucket="",
        )

    # ── Floor 3: Brightness (just blocks total darkness / pure glare) ────────
    brightness = _compute_brightness(gray)
    if brightness < ENROLL_MIN_BRIGHTNESS or brightness > ENROLL_MAX_BRIGHTNESS:
        return EnrollResult(
            accepted=False, reject_reason="BAD_LIGHTING",
            sharpness=sharpness, brightness=brightness, face_size=face_size,
            yaw=0.0, pitch=0.0, is_masked=False, bucket="",
        )

    # ── All floors passed — compute pose + mask for bucketing ────────────────
    yaw, pitch = 0.0, 0.0
    is_masked = False

    if face_obj is not None:
        # Pose estimation — use full-frame dimensions when landmarks are in full-frame
        # coordinate space (enrollment runs InsightFace on the full frame, so kps
        # are in full-frame pixel space, not crop space).
        if hasattr(face_obj, 'kps') and face_obj.kps is not None:
            pose_w = frame_w if frame_w is not None else w
            pose_h = frame_h if frame_h is not None else h
            yaw, pitch, _ = _estimate_pose(face_obj.kps, pose_w, pose_h)

        # Mask detection — uses both geometric ratio AND nose-mouth proximity
        is_masked = _detect_mask_for_enrollment(face_obj)

    # ── Bucket classification ─────────────────────────────────────────────────
    is_frontal = abs(yaw) <= frontal_yaw_threshold
    if is_frontal and not is_masked:
        bucket = BUCKET_FRONTAL_UNMASKED
    elif not is_frontal and not is_masked:
        bucket = BUCKET_ANGLED_UNMASKED
    elif is_frontal and is_masked:
        bucket = BUCKET_MASKED_FRONTAL
    else:
        bucket = BUCKET_MASKED_ANGLED

    return EnrollResult(
        accepted=True, reject_reason="",
        sharpness=sharpness, brightness=brightness, face_size=face_size,
        yaw=yaw, pitch=pitch, is_masked=is_masked, bucket=bucket,
    )


def _detect_mask_for_enrollment(face_obj) -> bool:
    """
    Mask detection for enrollment bucketing — tuned for full-frame high-res detection.

    In enrollment, InsightFace runs on the full camera frame at native resolution,
    so even masked faces are detected with high confidence (det_score 0.6+). The
    simple det_score < 0.5 fallback from the runtime path does NOT fire here.

    Instead we use two complementary geometry checks:

    1. Eye-to-mouth ratio: For normal faces, mouth is 1.0-1.5 eye-widths below
       eye center. When masked, InsightFace tends to cluster nose+mouth landmarks
       near the occlusion boundary (ratio < 0.75).

    2. Nose-to-mouth proximity: The vertical distance from nose tip (kps[2]) to
       mouth center should be about 0.5× eye-to-nose distance for a clear face.
       When masked, nose and mouth landmarks collapse together (ratio < 0.35).

    Either check firing is sufficient to classify as masked. This gives reasonable
    sensitivity even when InsightFace internally estimates landmark positions under
    a mask rather than directly observing them.
    """
    if not hasattr(face_obj, 'kps') or face_obj.kps is None:
        return False

    kps = face_obj.kps
    left_eye    = kps[0]
    right_eye   = kps[1]
    nose        = kps[2]
    mouth_left  = kps[3]
    mouth_right = kps[4]

    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < 5:
        return False

    eye_center   = (left_eye + right_eye) / 2
    mouth_center = (mouth_left + mouth_right) / 2

    # Check 1: Eye-to-mouth vs inter-eye distance
    eye_to_mouth = abs(mouth_center[1] - eye_center[1])
    ratio_mouth  = eye_to_mouth / eye_dist

    # Check 2: Nose-to-mouth vs eye-to-nose proximity
    eye_to_nose  = abs(nose[1] - eye_center[1])
    nose_to_mouth = abs(mouth_center[1] - nose[1])
    ratio_nose_mouth = (nose_to_mouth / eye_to_nose) if eye_to_nose > 2 else 1.0

    # Normal: ratio_mouth ~1.0-1.5, ratio_nose_mouth ~0.5-1.0
    # Masked: ratio_mouth < 0.75, ratio_nose_mouth < 0.30
    is_masked_ratio   = ratio_mouth < 0.75
    is_masked_proximity = ratio_nose_mouth < 0.30

    # Low det_score (< 0.65) is still a useful secondary signal at full resolution
    is_masked_score = hasattr(face_obj, 'det_score') and face_obj.det_score < 0.65

    return is_masked_ratio or is_masked_proximity or is_masked_score


def select_per_bucket(
    frames: list,
    n_per_bucket: int = 4,
    min_temporal_gap_s: float = 0.3,
) -> tuple:
    """
    Select up to n_per_bucket frames per condition bucket, ranked by sharpness,
    with a minimum temporal gap between selected frames (avoids near-duplicate frames).

    Args:
        frames: List of dicts, each must have keys:
                  "enroll_result" (EnrollResult), "timestamp" (float), + any other fields
        n_per_bucket: Max frames to keep per bucket.
        min_temporal_gap_s: Minimum seconds between selected frames within a bucket.

    Returns:
        (selected, bucket_counts, empty_buckets)
        - selected: list of frame dicts chosen for saving
        - bucket_counts: dict {bucket_name: n_selected}
        - empty_buckets: list of bucket names that got 0 frames
    """
    # Group by bucket
    buckets: dict = {b: [] for b in ALL_BUCKETS}
    for f in frames:
        b = f["enroll_result"].bucket
        if b in buckets:
            buckets[b].append(f)

    selected = []
    bucket_counts = {}

    for bucket_name in ALL_BUCKETS:
        candidates = buckets[bucket_name]
        # Sort by sharpness descending (best first)
        candidates.sort(key=lambda x: x["enroll_result"].sharpness, reverse=True)

        chosen = []
        for c in candidates:
            # Enforce minimum temporal gap within this bucket
            if all(abs(c["timestamp"] - s["timestamp"]) >= min_temporal_gap_s
                   for s in chosen):
                chosen.append(c)
            if len(chosen) >= n_per_bucket:
                break

        bucket_counts[bucket_name] = len(chosen)
        selected.extend(chosen)

    empty_buckets = [b for b, count in bucket_counts.items() if count == 0]
    return selected, bucket_counts, empty_buckets
