"""
quality.py
==========
Face quality gate for the gate monitor pipeline (Stage 7).

Runs lightweight checks on a face crop + landmarks BEFORE sending it to
ArcFace for recognition. A frame that fails the gate is skipped for
recognition but tracking continues normally.

Checks (in order, short-circuit on first failure):
  1. Face size — reject tiny distant faces
  2. Sharpness — variance of Laplacian (catches motion blur)
  3. Pose angle — yaw/pitch from 5-point landmarks via cv2.solvePnP
  4. Brightness — mean pixel intensity (reject too dark / blown out)

All thresholds are configurable in config.py with env overrides.

CPU-only, ~0.2ms per face crop. No GPU or deep model dependency.
"""

import cv2
import math
import numpy as np
from dataclasses import dataclass

from config import (
    MIN_FACE_PX, MIN_SHARPNESS, MAX_YAW_DEG, MAX_PITCH_DEG,
    MIN_BRIGHTNESS, MAX_BRIGHTNESS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QualityResult:
    """Result of the face quality gate assessment."""
    passed: bool            # True if all checks passed
    score: float            # Combined quality score 0.0-1.0
    reason: str             # "PASS", "TOO_SMALL", "TOO_BLURRY", "BAD_ANGLE", "BAD_LIGHTING"
    sharpness: float = 0.0  # Raw Laplacian variance
    face_size: int = 0      # min(width, height) of face crop
    yaw: float = 0.0        # Estimated yaw angle (degrees)
    pitch: float = 0.0      # Estimated pitch angle (degrees)
    brightness: float = 0.0 # Mean pixel intensity


# ─────────────────────────────────────────────────────────────────────────────
#  3D REFERENCE MODEL FOR HEAD POSE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
# Generic 3D face model points corresponding to 5-point landmarks:
# [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
# Units are arbitrary but proportions match an average adult face.
_MODEL_POINTS_5 = np.array([
    (-30.0,  -30.0, -30.0),   # Left eye
    ( 30.0,  -30.0, -30.0),   # Right eye
    (  0.0,    0.0,   0.0),   # Nose tip
    (-20.0,   30.0, -15.0),   # Left mouth corner
    ( 20.0,   30.0, -15.0),   # Right mouth corner
], dtype=np.float64)


def _estimate_pose(landmarks_2d: np.ndarray, img_w: int, img_h: int):
    """Estimate head pose (yaw, pitch, roll) from 5-point landmarks.

    Uses cv2.solvePnP with a generic 3D face model. Returns angles in degrees.
    This is CPU-only and takes ~0.1ms per call.

    Args:
        landmarks_2d: shape (5, 2) — 2D image coordinates of the 5 landmarks.
        img_w: Image width (for camera matrix estimation).
        img_h: Image height.

    Returns:
        (yaw, pitch, roll) in degrees, or (0, 0, 0) on failure.
    """
    if landmarks_2d is None or landmarks_2d.shape[0] < 5:
        return 0.0, 0.0, 0.0

    # Approximate camera intrinsics from image dimensions
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    pts_2d = landmarks_2d[:5].astype(np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_5, pts_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix to Euler angles
        # Using the convention: yaw (Y), pitch (X), roll (Z)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

    except cv2.error:
        return 0.0, 0.0, 0.0


def _compute_sharpness(gray_crop: np.ndarray) -> float:
    """Compute sharpness score as variance of Laplacian.

    Higher = sharper. Typical values:
    - Motion blur: 2-10
    - Acceptable: 15-50
    - Sharp: 50+
    """
    if gray_crop.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
    return float(lap.var())


def _compute_brightness(gray_crop: np.ndarray) -> float:
    """Compute mean pixel intensity of the face crop.

    Returns value in [0, 255]. Healthy range: 40-220.
    """
    if gray_crop.size == 0:
        return 128.0
    return float(np.mean(gray_crop))


def assess_quality(face_crop_bgr: np.ndarray,
                   face_obj=None,
                   min_face_px: int = None,
                   min_sharpness: float = None,
                   max_yaw: int = None,
                   max_pitch: int = None,
                   min_brightness: int = None,
                   max_brightness: int = None,
                   frame_w: int = None,
                   frame_h: int = None) -> QualityResult:
    """Assess face quality for recognition suitability.

    Runs lightweight CPU checks in order of computational cost.
    Short-circuits on first failure.

    Args:
        face_crop_bgr: BGR face crop from the frame.
        face_obj: InsightFace face object with .kps landmarks (optional).
                  If None, pose check is skipped.
        min_face_px: Override for MIN_FACE_PX config.
        min_sharpness: Override for MIN_SHARPNESS config.
        max_yaw: Override for MAX_YAW_DEG config.
        max_pitch: Override for MAX_PITCH_DEG config.
        min_brightness: Override for MIN_BRIGHTNESS config.
        max_brightness: Override for MAX_BRIGHTNESS config.
        frame_w: Width of the coordinate space that face_obj.kps is in.
        frame_h: Height of the coordinate space that face_obj.kps is in.

    Returns:
        QualityResult with pass/fail verdict, combined score, and reason code.
    """
    # Apply defaults from config
    _min_face = min_face_px if min_face_px is not None else MIN_FACE_PX
    _min_sharp = min_sharpness if min_sharpness is not None else MIN_SHARPNESS
    _max_yaw = max_yaw if max_yaw is not None else MAX_YAW_DEG
    _max_pitch = max_pitch if max_pitch is not None else MAX_PITCH_DEG
    _min_bright = min_brightness if min_brightness is not None else MIN_BRIGHTNESS
    _max_bright = max_brightness if max_brightness is not None else MAX_BRIGHTNESS

    h, w = face_crop_bgr.shape[:2]
    face_size = min(w, h)
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    # ── Check 1: Face size ──────────────────────────────────────────────
    if face_size < _min_face:
        return QualityResult(
            passed=False, score=0.0, reason="TOO_SMALL",
            face_size=face_size,
        )

    # ── Check 2: Sharpness ──────────────────────────────────────────────
    sharpness = _compute_sharpness(gray)
    if sharpness < _min_sharp:
        return QualityResult(
            passed=False, score=0.0, reason="TOO_BLURRY",
            sharpness=sharpness, face_size=face_size,
        )

    # ── Check 3: Pose angle ─────────────────────────────────────────────
    yaw, pitch = 0.0, 0.0
    if face_obj is not None and hasattr(face_obj, 'kps') and face_obj.kps is not None:
        pose_w = frame_w if frame_w is not None else w
        pose_h = frame_h if frame_h is not None else h
        yaw, pitch, _ = _estimate_pose(face_obj.kps, pose_w, pose_h)
        if abs(yaw) > _max_yaw or abs(pitch) > _max_pitch:
            return QualityResult(
                passed=False, score=0.0, reason="BAD_ANGLE",
                sharpness=sharpness, face_size=face_size,
                yaw=yaw, pitch=pitch,
            )

    # ── Check 4: Brightness ─────────────────────────────────────────────
    brightness = _compute_brightness(gray)
    if brightness < _min_bright or brightness > _max_bright:
        return QualityResult(
            passed=False, score=0.0, reason="BAD_LIGHTING",
            sharpness=sharpness, face_size=face_size,
            yaw=yaw, pitch=pitch, brightness=brightness,
        )

    # ── All checks passed — compute combined score ──────────────────────
    # Normalize each metric to [0, 1] and combine
    size_norm = min(1.0, face_size / 120.0)        # 120px = perfect size
    sharp_norm = min(1.0, sharpness / 100.0)       # 100 = very sharp
    angle_norm = 1.0 - (max(abs(yaw), abs(pitch)) / 90.0)  # 0° = perfect
    angle_norm = max(0.0, angle_norm)

    # Brightness score: peak at 128 (mid-gray), drops off toward 0 and 255
    bright_norm = 1.0 - abs(brightness - 128.0) / 128.0
    bright_norm = max(0.0, bright_norm)

    score = (sharp_norm * 0.4 + size_norm * 0.3 +
             angle_norm * 0.2 + bright_norm * 0.1)
    score = max(0.0, min(1.0, score))

    return QualityResult(
        passed=True, score=score, reason="PASS",
        sharpness=sharpness, face_size=face_size,
        yaw=yaw, pitch=pitch, brightness=brightness,
    )
