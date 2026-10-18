"""
tools/probe_plate_size.py
=========================
Measure licence-plate pixel size, angle, blur, smear, and IR glare
from recorded clips or live camera snapshots.

Runs a plate detector on sample frames and reports statistics to help
set min_plate_px and assess camera suitability.

Usage:
    python tools/probe_plate_size.py --source data/clips/cctv_gate_20261015.mp4
    python tools/probe_plate_size.py --source data/clips/ --sample-frames 100

Output:
    - Per-plate measurements: width_px, height_px, aspect_ratio, blur_score,
      skew_angle_deg, detector_confidence
    - Summary statistics: min/max/median/p10/p90 for each metric
    - Recommendation for min_plate_px threshold
    - Saves annotated sample frames to data/review/plate_probe/
"""

import argparse
import glob
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np


@dataclass
class PlateMeasurement:
    """Measurement of a single detected plate."""
    frame_idx: int
    source_file: str
    x1: int
    y1: int
    x2: int
    y2: int
    width_px: int
    height_px: int
    aspect_ratio: float
    blur_score: float        # Laplacian variance (higher = sharper)
    skew_angle_deg: float    # Estimated rotation angle
    detector_conf: float
    brightness_mean: float
    ir_glare_score: float    # Fraction of near-white pixels (potential IR glare)


def compute_blur_score(gray_crop: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    if gray_crop.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
    return float(lap.var())


def compute_skew_angle(gray_crop: np.ndarray) -> float:
    """Estimate plate skew using minAreaRect on edges."""
    if gray_crop.size < 100:
        return 0.0
    edges = cv2.Canny(gray_crop, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 10:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    # Normalize angle to [-45, 45]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return round(angle, 1)


def compute_ir_glare_score(gray_crop: np.ndarray) -> float:
    """Fraction of near-white pixels (brightness > 240), indicating IR glare."""
    if gray_crop.size == 0:
        return 0.0
    return float(np.mean(gray_crop > 240))


def measure_plates_in_frame(
    frame: np.ndarray,
    frame_idx: int,
    source_file: str,
    plate_boxes: list,
) -> List[PlateMeasurement]:
    """Measure all detected plates in a single frame.

    Args:
        frame: BGR frame.
        frame_idx: Frame number for reference.
        source_file: Source filename.
        plate_boxes: List of (x1, y1, x2, y2, confidence) tuples.

    Returns:
        List of PlateMeasurement objects.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    measurements = []

    for (x1, y1, x2, y2, conf) in plate_boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        w = x2 - x1
        h = y2 - y1
        if w < 5 or h < 5:
            continue

        crop_gray = gray[y1:y2, x1:x2]

        m = PlateMeasurement(
            frame_idx=frame_idx,
            source_file=os.path.basename(source_file),
            x1=x1, y1=y1, x2=x2, y2=y2,
            width_px=w,
            height_px=h,
            aspect_ratio=round(w / h, 2) if h > 0 else 0.0,
            blur_score=round(compute_blur_score(crop_gray), 1),
            skew_angle_deg=compute_skew_angle(crop_gray),
            detector_conf=round(float(conf), 3),
            brightness_mean=round(float(crop_gray.mean()), 1),
            ir_glare_score=round(compute_ir_glare_score(crop_gray), 3),
        )
        measurements.append(m)

    return measurements


def summarize_measurements(measurements: List[PlateMeasurement]) -> dict:
    """Compute summary statistics from all measurements."""
    if not measurements:
        return {"count": 0, "recommendation": "No plates detected"}

    widths = [m.width_px for m in measurements]
    heights = [m.height_px for m in measurements]
    blurs = [m.blur_score for m in measurements]
    aspects = [m.aspect_ratio for m in measurements]
    skews = [abs(m.skew_angle_deg) for m in measurements]
    confs = [m.detector_conf for m in measurements]
    ir_scores = [m.ir_glare_score for m in measurements]

    def _stats(values, name):
        arr = np.array(values)
        return {
            "name": name,
            "count": len(arr),
            "min": round(float(np.min(arr)), 1),
            "max": round(float(np.max(arr)), 1),
            "median": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "mean": round(float(np.mean(arr)), 1),
        }

    summary = {
        "count": len(measurements),
        "width_px": _stats(widths, "width_px"),
        "height_px": _stats(heights, "height_px"),
        "aspect_ratio": _stats(aspects, "aspect_ratio"),
        "blur_score": _stats(blurs, "blur_score"),
        "skew_angle_abs": _stats(skews, "skew_angle_abs"),
        "detector_conf": _stats(confs, "detector_conf"),
        "ir_glare_score": _stats(ir_scores, "ir_glare_score"),
    }

    # Recommendation
    median_w = float(np.median(widths))
    p10_w = float(np.percentile(widths, 10))
    if p10_w >= 120:
        rec = f"GOOD — p10 width={p10_w:.0f}px. Set min_plate_px=80."
    elif p10_w >= 80:
        rec = (f"MARGINAL — p10 width={p10_w:.0f}px. "
               f"Set min_plate_px=60. Some plates may be unreadable.")
    else:
        rec = (f"POOR — p10 width={p10_w:.0f}px. "
               f"Camera repositioning or zoom needed. "
               f"Consider narrower FoV or moving camera closer to the gate.")

    summary["recommendation"] = rec
    summary["suggested_min_plate_px"] = max(40, int(p10_w * 0.6))

    return summary


def probe_with_detector(
    source_path: str,
    sample_frames: int = 200,
    output_dir: Optional[str] = None,
) -> dict:
    """Run plate detection on sample frames and measure plates.

    Uses open-image-models plate detector if available, otherwise
    falls back to contour-based heuristic detection.

    Args:
        source_path: Video file or directory of videos.
        sample_frames: Number of frames to sample.
        output_dir: Where to save annotated frames.

    Returns:
        Summary dict with stats and recommendation.
    """
    # Collect video files
    if os.path.isdir(source_path):
        files = sorted(
            glob.glob(os.path.join(source_path, "*.mp4")) +
            glob.glob(os.path.join(source_path, "*.avi"))
        )
        if not files:
            return {"error": f"No video files found in {source_path}"}
    else:
        files = [source_path]

    # Try to load the plate detector
    detector = None
    try:
        from open_image_models import LicensePlateDetector
        detector = LicensePlateDetector(
            detection_model="yolo-v9-t-384-license-plate-end2end"
        )
        print("  Using open-image-models plate detector")
    except ImportError:
        print("  ⚠ open-image-models not installed, using contour heuristic")
    except Exception as e:
        print(f"  ⚠ Plate detector init failed: {e}, using contour heuristic")

    all_measurements = []
    frames_per_file = max(1, sample_frames // len(files))

    for filepath in files:
        print(f"\n  Processing: {os.path.basename(filepath)}")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"    ✗ Cannot open {filepath}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 10000  # fallback for streams

        # Sample evenly
        step = max(1, total // frames_per_file)
        sampled = 0

        for fidx in range(0, total, step):
            if sampled >= frames_per_file:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break

            # Detect plates
            plate_boxes = []
            if detector is not None:
                try:
                    results = detector.predict(frame)
                    # Results are supervision Detections
                    if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                        for i in range(len(results.xyxy)):
                            x1, y1, x2, y2 = results.xyxy[i]
                            conf = results.confidence[i] if results.confidence is not None else 0.5
                            plate_boxes.append((x1, y1, x2, y2, conf))
                except Exception as e:
                    print(f"    ⚠ Detection error at frame {fidx}: {e}")
            else:
                # Contour heuristic fallback
                plate_boxes = _heuristic_plate_detect(frame)

            if plate_boxes:
                ms = measure_plates_in_frame(frame, fidx, filepath, plate_boxes)
                all_measurements.extend(ms)

                # Save annotated sample frames
                if output_dir and len(all_measurements) <= 50:
                    _save_annotated(frame, ms, output_dir, fidx, filepath)

            sampled += 1

        cap.release()
        print(f"    Sampled {sampled} frames, found {len([m for m in all_measurements if m.source_file == os.path.basename(filepath)])} plates")

    # Summary
    summary = summarize_measurements(all_measurements)
    summary["sources"] = [os.path.basename(f) for f in files]
    summary["frames_sampled"] = sample_frames

    # Save raw measurements
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, "plate_measurements.json")
        with open(raw_path, "w") as f:
            json.dump(
                {"measurements": [asdict(m) for m in all_measurements],
                 "summary": summary},
                f, indent=2
            )
        print(f"\n  Raw data saved to: {raw_path}")

    return summary


def _heuristic_plate_detect(frame: np.ndarray) -> list:
    """Simple contour-based plate candidate detection (fallback).

    Looks for rectangular contours with plate-like aspect ratios.
    Not accurate — use only when the ML detector is unavailable.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    h_frame, w_frame = frame.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        area = w * h
        # Plate-like: aspect 1.5-6, reasonable size
        if 1.5 < aspect < 6.0 and area > 500 and w > 30:
            # Not too large (not the whole frame)
            if w < w_frame * 0.5 and h < h_frame * 0.3:
                plates.append((x, y, x + w, y + h, 0.3))

    return plates[:20]  # Limit candidates


def _save_annotated(frame, measurements, output_dir, fidx, source):
    """Save annotated frame with plate boxes drawn."""
    os.makedirs(output_dir, exist_ok=True)
    annotated = frame.copy()
    for m in measurements:
        color = (0, 255, 0) if m.width_px >= 120 else (0, 165, 255) if m.width_px >= 80 else (0, 0, 255)
        cv2.rectangle(annotated, (m.x1, m.y1), (m.x2, m.y2), color, 2)
        label = f"{m.width_px}px blur={m.blur_score:.0f}"
        cv2.putText(annotated, label, (m.x1, m.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    basename = os.path.splitext(os.path.basename(source))[0]
    out_path = os.path.join(output_dir, f"{basename}_f{fidx:05d}.jpg")
    cv2.imwrite(out_path, annotated)


def main():
    parser = argparse.ArgumentParser(
        description="Probe plate sizes from recorded clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Video file or directory of clips")
    parser.add_argument("--sample-frames", type=int, default=200,
                        help="Number of frames to sample (default: 200)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "review" / "plate_probe"),
                        help="Output dir for annotated frames + data")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PLATE SIZE PROBE")
    print(f"{'='*60}")

    summary = probe_with_detector(
        args.source,
        sample_frames=args.sample_frames,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    if "error" in summary:
        print(f"  ✗ {summary['error']}")
        sys.exit(1)

    if summary["count"] == 0:
        print("  No plates detected in sampled frames.")
        print("  This may indicate:")
        print("    - No vehicles in the clips")
        print("    - Camera angle/distance unsuitable")
        print("    - Clips are too dark or blurry")
        sys.exit(0)

    print(f"  Plates detected: {summary['count']}")
    for key in ["width_px", "height_px", "aspect_ratio", "blur_score",
                 "skew_angle_abs", "ir_glare_score"]:
        if key in summary:
            s = summary[key]
            print(f"  {s['name']:20s}  min={s['min']:7.1f}  "
                  f"p10={s['p10']:7.1f}  median={s['median']:7.1f}  "
                  f"p90={s['p90']:7.1f}  max={s['max']:7.1f}")

    print(f"\n  📋 {summary['recommendation']}")
    print(f"  Suggested min_plate_px: {summary.get('suggested_min_plate_px', '?')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
