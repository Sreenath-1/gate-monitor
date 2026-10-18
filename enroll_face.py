"""
enroll_face.py
==============
Bucket-aware interactive face enrollment tool (Stage 7+).

Key changes from the original:
  - Replaces the strict frontality filter with enroll_quality.assess_enroll_quality()
    which has NO angle or occlusion rejection — those are exactly what we need to store.
  - Classifies each accepted frame into 4 condition buckets:
      frontal_unmasked | angled_unmasked | masked_frontal | masked_angled
  - Keeps top-N frames per bucket by sharpness score (default N=4, --per-bucket)
    instead of one global top-8 ranking (which always picks frontal/unmasked shots).
  - Warns the user at the end if any bucket is empty so they can re-run and specifically
    fill the gap.
  - Saves a bucket_metadata.json alongside the images so build_known_encodings()
    can build separate full-face and periocular templates correctly.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_RTSP_URL, IST
from stream import FrameSource
from face_id import _load_insightface, _face_app
from enroll_quality import (
    assess_enroll_quality, select_per_bucket,
    ALL_BUCKETS, BUCKET_FRONTAL_UNMASKED, BUCKET_ANGLED_UNMASKED,
    BUCKET_MASKED_FRONTAL, BUCKET_MASKED_ANGLED,
)


# ── Human-readable bucket labels for the live display ────────────────────────
_BUCKET_LABELS = {
    BUCKET_FRONTAL_UNMASKED: "Frontal (no mask)",
    BUCKET_ANGLED_UNMASKED:  "Angled  (no mask)",
    BUCKET_MASKED_FRONTAL:   "Frontal (MASKED)",
    BUCKET_MASKED_ANGLED:    "Angled  (MASKED)",
}
_BUCKET_COLORS = {
    BUCKET_FRONTAL_UNMASKED: (0, 220, 0),      # Green
    BUCKET_ANGLED_UNMASKED:  (0, 180, 255),    # Orange
    BUCKET_MASKED_FRONTAL:   (255, 180, 0),    # Blue-ish
    BUCKET_MASKED_ANGLED:    (200, 0, 200),    # Purple
}


def main():
    parser = argparse.ArgumentParser(
        description="Bucket-aware interactive face enrollment tool."
    )
    parser.add_argument("--name", type=str, default=None,
                        help="(DEPRECATED) Alias for --employee-id for backward compat")
    parser.add_argument("--employee-id", type=str, default=None,
                        help="Unique employee identifier (e.g. badge number, 4-digit ID). "
                             "This becomes the directory name in known_faces/.")
    parser.add_argument("--display-name", type=str, default=None,
                        help="Human-readable display name (e.g. 'Alice V.'). "
                             "Optional — if not provided, employee_id is used.")
    parser.add_argument("--duration", type=int, default=90,
                        help="Capture duration in seconds (default: 90)")
    parser.add_argument("--source", type=str, default="rtsp",
                        help="Video source: 'rtsp', a file path, or webcam index")
    parser.add_argument("--per-bucket", type=int, default=4,
                        help="Max frames to save per condition bucket (default: 4)")
    parser.add_argument("--frontal-yaw", type=float, default=20.0,
                        help="Yaw threshold for frontal vs angled buckets (default: 20°). "
                             "Raise to 25-30° if camera is mounted high/overhead.")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live preview window (for headless use)")
    args = parser.parse_args()

    # Resolve employee_id: --employee-id takes precedence, --name is backward compat
    employee_id = args.employee_id or args.name
    if not employee_id:
        print("ERROR: --employee-id is required (or --name for backward compat).")
        print("  Example: python enroll_face.py --employee-id 0042 --display-name 'Alice V.'")
        return
    display_name = args.display_name or employee_id

    source_url = DEFAULT_RTSP_URL if args.source.lower() == "rtsp" else args.source

    print(f"\n{'='*60}")
    print(f"  FACE ENROLLMENT TOOL  (bucket-aware, Stage 7+)")
    print(f"{'='*60}")
    print(f"  Employee ID  : {employee_id}")
    print(f"  Display name : {display_name}")
    print(f"  Duration     : {args.duration} seconds")
    print(f"  Source       : {source_url}")
    print(f"  Per bucket   : {args.per_bucket} frames")
    print(f"  Frontal yaw  : ≤ {args.frontal_yaw}°")
    print(f"{'='*60}\n")
    print(f"  Condition buckets that will be captured:")
    for b in ALL_BUCKETS:
        print(f"    [{b}]")
    print()

    if not _load_insightface():
        print("Error: Could not load InsightFace models.")
        return

    print("Initializing camera stream (full resolution)...")
    try:
        source = FrameSource(source_url, resize_height=0)
    except Exception as e:
        print(f"Error opening source: {e}")
        return

    print(f"\n>>> Walk through the gate for {args.duration} seconds.")
    print(f"    Enrolling employee: {display_name} (ID: {employee_id})")
    print("    Do all 4 of these during capture:")
    print("    1) Walk normally without mask (frontal + angled)")
    print("    2) Put mask on, walk past at normal pace")
    print("    3) Walk fast with mask on")
    print("    4) Briefly face camera directly both masked and unmasked")
    print()

    start_time = time.time()
    accepted_frames = []   # list of frame dicts that passed the loose floor
    rejected_counts = {"TOO_SMALL": 0, "TOO_BLURRY": 0, "BAD_LIGHTING": 0, "no_face": 0}
    # Running bucket counter for live display
    live_bucket_counts = {b: 0 for b in ALL_BUCKETS}

    while True:
        elapsed = time.time() - start_time
        remaining = args.duration - elapsed
        if remaining <= 0:
            break

        if source.connection_lost:
            print("Camera connection lost.")
            break

        ret, frame = source.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import face_id
        faces = face_id._face_app.get(frame_rgb)

        display_frame = frame.copy() if not args.no_display else None
        if display_frame is not None:
            h_d, w_d = display_frame.shape[:2]
            display_scale = min(1.0, 960 / max(w_d, 1))
            if display_scale < 1.0:
                display_frame = cv2.resize(
                    display_frame,
                    (int(w_d * display_scale), int(h_d * display_scale))
                )

        if faces:
            # Pick the face with highest detection score
            best_face = max(faces, key=lambda f: f.det_score)
            score = best_face.det_score

            # Only consider faces with a minimum detection confidence
            if score >= 0.5:
                x1, y1, x2, y2 = map(int, best_face.bbox)
                # Build face crop for quality assessment
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                h_f, w_f = frame_bgr.shape[:2]
                fx1, fy1 = max(0, x1), max(0, y1)
                fx2, fy2 = min(w_f, x2), min(h_f, y2)
                face_crop_bgr = frame_bgr[fy1:fy2, fx1:fx2]

                if face_crop_bgr.size > 0:
                    enroll_result = assess_enroll_quality(
                        face_crop_bgr, best_face,
                        frontal_yaw_threshold=args.frontal_yaw,
                        frame_w=w_f, frame_h=h_f,  # full-frame dims for correct pose
                    )

                    if enroll_result.accepted:
                        live_bucket_counts[enroll_result.bucket] += 1
                        accepted_frames.append({
                            "enroll_result": enroll_result,
                            "frame_rgb": frame_rgb,
                            "bbox": best_face.bbox,
                            "face_obj": best_face,
                            "timestamp": time.time(),
                            "det_score": score,
                        })

                        if display_frame is not None:
                            color = _BUCKET_COLORS[enroll_result.bucket]
                            label = _BUCKET_LABELS[enroll_result.bucket]
                            dx1 = int(x1 * display_scale)
                            dy1 = int(y1 * display_scale)
                            dx2 = int(x2 * display_scale)
                            dy2 = int(y2 * display_scale)
                            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                            cv2.putText(display_frame, label,
                                        (dx1, max(dy1 - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            cv2.putText(display_frame,
                                        f"yaw={enroll_result.yaw:.0f}° "
                                        f"sharp={enroll_result.sharpness:.0f}",
                                        (dx1, dy2 + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    else:
                        rejected_counts[enroll_result.reject_reason] = (
                            rejected_counts.get(enroll_result.reject_reason, 0) + 1
                        )
                        if display_frame is not None:
                            dx1 = int(x1 * display_scale)
                            dy1 = int(y1 * display_scale)
                            dx2 = int(x2 * display_scale)
                            dy2 = int(y2 * display_scale)
                            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 0, 200), 1)
                            cv2.putText(display_frame,
                                        f"Skip: {enroll_result.reject_reason}",
                                        (dx1, max(dy1 - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
        else:
            rejected_counts["no_face"] += 1

        if display_frame is not None:
            # Overlay: enrolled name + timer
            cv2.putText(display_frame, f"Enrolling: {display_name} (ID: {employee_id})",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {remaining:.0f}s",
                        (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Bucket counters on right side
            y_off = 36
            for b in ALL_BUCKETS:
                count = live_bucket_counts[b]
                label = f"{_BUCKET_LABELS[b]}: {count}"
                color = _BUCKET_COLORS[b] if count > 0 else (120, 120, 120)
                cv2.putText(display_frame, label,
                            (display_frame.shape[1] - 310, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_off += 24

            cv2.imshow("Enrollment Tool", display_frame)
            if cv2.waitKey(1) == 27:
                print("Aborted by user.")
                break

    if not args.no_display:
        cv2.destroyAllWindows()
    source.release()

    # ── Selection phase ───────────────────────────────────────────────────────
    print("\n--- Capture Complete ---")
    print(f"Frames accepted by quality floor: {len(accepted_frames)}")
    print(f"Rejections: {rejected_counts}")
    print(f"\nRaw frames per bucket:")
    for b in ALL_BUCKETS:
        print(f"  [{b}]: {live_bucket_counts[b]} frames")

    if not accepted_frames:
        print("\nNo usable frames captured. Please re-run.")
        return

    selected, bucket_counts, empty_buckets = select_per_bucket(
        accepted_frames,
        n_per_bucket=args.per_bucket,
        min_temporal_gap_s=0.3,
    )

    print(f"\nSelected {len(selected)} frames (up to {args.per_bucket} per bucket):")
    for b in ALL_BUCKETS:
        print(f"  [{b}]: {bucket_counts[b]} frames selected")

    # ── Warnings for empty buckets ────────────────────────────────────────────
    if empty_buckets:
        print(f"\n⚠  EMPTY BUCKETS WARNING:")
        for b in empty_buckets:
            if b == BUCKET_FRONTAL_UNMASKED:
                print(f"  [{b}] — No unmasked frontal frames captured.")
                print(f"         Re-run and face the camera directly without mask for 10s.")
            elif b == BUCKET_ANGLED_UNMASKED:
                print(f"  [{b}] — No unmasked angled frames captured.")
                print(f"         Re-run and walk past the camera without mask.")
            elif b == BUCKET_MASKED_FRONTAL:
                print(f"  [{b}] — No masked frontal frames captured.")
                print(f"         Re-run wearing mask, pause and face camera directly.")
            elif b == BUCKET_MASKED_ANGLED:
                print(f"  [{b}] — No masked angled frames captured.")
                print(f"         Re-run wearing mask, walk past the camera at normal pace.")

    if len(selected) < 3:
        print(f"\nFewer than 3 total frames selected ({len(selected)}). Please re-run.")
        return

    # ── Duplicate enrollment guard ─────────────────────────────────────────────
    # Before saving, cross-check embeddings against ALL other enrolled identities
    from config import GATE_DUPLICATE_ENROLL_THRESHOLD
    from face_id import build_known_encodings

    print("\nRunning duplicate enrollment check against existing gallery...")
    existing_gallery = build_known_encodings(
        str(Path(__file__).parent / "known_faces")
    )

    if existing_gallery:
        # Compute average embedding from the selected frames
        new_embs = []
        for cf in selected:
            face_obj = cf.get("face_obj")
            frame_rgb = cf["frame_rgb"]
            if face_obj is not None and hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
                emb = face_obj.embedding / np.linalg.norm(face_obj.embedding)
                new_embs.append(emb)

        if new_embs:
            avg_new_emb = np.mean(new_embs, axis=0)
            avg_new_emb = avg_new_emb / np.linalg.norm(avg_new_emb)

            for existing_id, entry in existing_gallery.items():
                # Skip self (if re-enrolling the same person)
                if existing_id == employee_id:
                    continue
                existing_full = entry.get("full")
                if existing_full is not None:
                    sim = float(np.dot(avg_new_emb, existing_full))
                    if sim > GATE_DUPLICATE_ENROLL_THRESHOLD:
                        print(f"\n{'!'*60}")
                        print(f"  ⚠  DUPLICATE ENROLLMENT WARNING")
                        print(f"{'!'*60}")
                        print(f"  This face strongly matches already-enrolled employee:")
                        print(f"    Existing ID : {existing_id}")
                        print(f"    Similarity  : {sim:.3f} (threshold: {GATE_DUPLICATE_ENROLL_THRESHOLD})")
                        print(f"")
                        print(f"  This could be:")
                        print(f"    1. The same person being enrolled twice (operator error)")
                        print(f"    2. A family member or very similar-looking person")
                        print(f"")
                        print(f"  Enrollment HALTED. If this is intentional, raise")
                        print(f"  GATE_DUPLICATE_ENROLL_THRESH in .env above {sim:.3f}")
                        print(f"{'!'*60}\n")
                        return
                    elif sim > GATE_DUPLICATE_ENROLL_THRESHOLD - 0.10:
                        print(f"  Note: moderate similarity ({sim:.3f}) to existing "
                              f"employee {existing_id} — watch for confusion at runtime")

        print("  ✓ No duplicate enrollments detected.")
    else:
        print("  (No existing enrollments to check against — first enrollment.)")

    # ── Save phase ────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "known_faces" / employee_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(selected)} frames for {display_name} (ID: {employee_id}):")

    saved_metadata = []
    selected.sort(key=lambda x: x["timestamp"])

    for idx, cf in enumerate(selected):
        er = cf["enroll_result"]
        frame_rgb = cf["frame_rgb"]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h_f, w_f = frame_bgr.shape[:2]

        x1, y1, x2, y2 = map(int, cf["bbox"])
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * 0.8)
        pad_y = int(bh * 0.8)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w_f, x2 + pad_x)
        cy2 = min(h_f, y2 + pad_y)
        crop = frame_bgr[cy1:cy2, cx1:cx2]

        dt_str = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        filename = f"{employee_id}_{dt_str}_{idx+1}.jpg"

        crop_path = out_dir / filename
        raw_path  = raw_dir / filename
        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(raw_path), frame_bgr)

        print(f"  [{idx+1}/{len(selected)}] {filename}"
              f" | bucket={er.bucket}"
              f" | yaw={er.yaw:.1f}° masked={er.is_masked}"
              f" | sharp={er.sharpness:.0f} bright={er.brightness:.0f}"
              f" | det={cf['det_score']:.3f}")

        saved_metadata.append({
            "filename":   filename,
            "bucket":     er.bucket,
            "yaw":        round(float(er.yaw), 2),
            "pitch":      round(float(er.pitch), 2),
            "is_masked":  bool(er.is_masked),
            "sharpness":  round(float(er.sharpness), 2),
            "brightness": round(float(er.brightness), 2),
            "det_score":  round(float(cf["det_score"]), 4),
        })

    # ── Write or Merge bucket_metadata.json ───────────────────────────────────
    meta_path = out_dir / "bucket_metadata.json"
    
    # Load existing metadata if it exists
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                existing_meta = json.load(f)
            
            # Merge frames
            existing_frames = existing_meta.get("frames", [])
            existing_frames.extend(saved_metadata)
            saved_metadata = existing_frames
            
            # Merge bucket counts
            existing_counts = existing_meta.get("bucket_counts", {})
            for b in ALL_BUCKETS:
                bucket_counts[b] += existing_counts.get(b, 0)
                
            print(f"\nAppended to existing gallery. Total frames now: {len(saved_metadata)}")
        except Exception as e:
            print(f"Warning: could not merge existing metadata: {e}")

    metadata = {
        "employee_id":   employee_id,
        "display_name":  display_name,
        "name":          display_name,  # backward compat
        "enrolled_at":   datetime.now(IST).isoformat(),
        "bucket_counts": bucket_counts,
        "frames":        saved_metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBucket metadata saved: {meta_path}")
    print(f"\nSuccessfully enrolled {display_name} (ID: {employee_id}): "
          f"{len(saved_metadata)} frames across "
          f"{sum(1 for c in bucket_counts.values() if c > 0)}/{len(ALL_BUCKETS)} buckets.")

    if empty_buckets:
        print(f"\n⚠  Still missing buckets: {empty_buckets}")
        print(f"   Run enroll_face.py again — it will ADD frames to the existing gallery.")
        print(f"   (Existing files are not deleted, new ones are appended)")
    else:
        print(f"\n✓  All 4 condition buckets populated. Gallery is complete.")

    print(f"\nDirectory: {out_dir}")
    print("Run: .\.python\python.exe calibrate_thresholds.py  (to get suggested thresholds)")
    print("Then: .\.python\python.exe main.py --source rtsp")


if __name__ == "__main__":
    main()
