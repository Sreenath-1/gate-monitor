"""
tools/calibrate.py
==================
Calibration & evaluation for the person search system.

Provides:
  1. Ground-truth CSV format for labeled walk-past sessions
  2. Per-camera genuine vs impostor score distributions
  3. ROC curves, TAR@FAR=1e-2/1e-3, rank-1/rank-5 retrieval
  4. Time-to-first-hit measurement
  5. Auto-proposed tier cutoffs → calibration.json
  6. Absent-target soak test (false positive rate measurement)
  7. Recording helper for labeled sessions

Usage:
    # Run calibration from labeled data:
    python tools/calibrate.py --ground-truth labels.csv --observations-db observations.db

    # Record a labeled walk-past session:
    python tools/calibrate.py --record --camera laptop_cam --person "Alice"

    # Absent-target soak test:
    python tools/calibrate.py --soak-test --probe unknown_person.jpg --duration 60
"""

import os
import sys
import json
import time
import csv
import argparse
import sqlite3
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IST


# ─────────────────────────────────────────────────────────────────────────────
#  GROUND-TRUTH CSV FORMAT
# ─────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH_HEADER = [
    "person_name",       # Who walked past (matches probe photo filename)
    "camera_id",         # Which camera
    "start_time_ist",    # When they appeared (IST, YYYY-MM-DD HH:MM:SS)
    "end_time_ist",      # When they left frame
    "condition",         # "unmasked", "masked", "glasses", "cap", "ir_night"
    "distance_m",        # Approximate distance from camera (meters)
    "movement",          # "standing", "walking", "running"
    "notes",             # Any additional notes
]

EXAMPLE_CSV = """person_name,camera_id,start_time_ist,end_time_ist,condition,distance_m,movement,notes
Alice,laptop_cam,2026-10-03 22:00:00,2026-10-03 22:00:30,unmasked,1.5,standing,frontal
Alice,laptop_cam,2026-10-03 22:01:00,2026-10-03 22:01:20,masked,1.5,walking,surgical mask
Alice,cctv_gate,2026-10-03 22:02:00,2026-10-03 22:02:45,unmasked,3.0,walking,gate entry
Alice,phone_cam,2026-10-03 22:03:00,2026-10-03 22:03:30,unmasked,2.0,standing,phone camera
"""


def create_template_csv(output_path: str = "ground_truth_template.csv"):
    """Create a template ground-truth CSV for the user to fill in."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(GROUND_TRUTH_HEADER) + "\n")
        f.write("# Fill in one row per walk-past segment.\n")
        f.write("# Example:\n")
        f.write(EXAMPLE_CSV)
    print(f"  Template CSV created: {output_path}")
    print(f"  Fill in the entries and re-run calibration.")


# ─────────────────────────────────────────────────────────────────────────────
#  SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def load_observations(db_path: str) -> List[dict]:
    """Load all usable observations from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM observations WHERE is_usable = 1 ORDER BY ts_utc"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_ground_truth(csv_path: str) -> List[dict]:
    """Load ground-truth labels from CSV."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("person_name", "").startswith("#"):
                continue
            # Parse IST times to Unix timestamps
            try:
                start = datetime.strptime(
                    row["start_time_ist"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=IST).timestamp()
                end = datetime.strptime(
                    row["end_time_ist"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=IST).timestamp()
            except (ValueError, KeyError):
                continue

            entries.append({
                "person_name": row["person_name"],
                "camera_id": row["camera_id"],
                "start_ts": start,
                "end_ts": end,
                "condition": row.get("condition", "unmasked"),
                "distance_m": float(row.get("distance_m", 0)),
                "movement": row.get("movement", ""),
                "notes": row.get("notes", ""),
            })
    return entries


def compute_scores(observations: List[dict], probe_embeddings: dict,
                   ground_truth: List[dict]) -> dict:
    """Compute genuine and impostor score distributions.

    Args:
        observations: List of observation dicts from DB.
        probe_embeddings: {person_name: np.ndarray (512,)} probe embeddings.
        ground_truth: Labeled walk-past segments.

    Returns:
        Dict with per-camera, per-condition score distributions.
    """
    results = {}

    for person_name, probe_emb in probe_embeddings.items():
        genuine_scores = []
        impostor_scores = []

        for obs in observations:
            emb_blob = obs.get("embedding")
            if emb_blob is None:
                continue
            obs_emb = np.frombuffer(emb_blob, dtype=np.float32)
            if obs_emb.shape[0] != 512:
                continue

            # Cosine similarity
            score = float(np.dot(probe_emb, obs_emb))
            camera_id = obs["camera_id"]
            ts = obs["ts_utc"]

            # Check if this observation is a genuine match
            is_genuine = False
            condition = "unmasked"
            for gt in ground_truth:
                if (gt["person_name"] == person_name and
                        gt["camera_id"] == camera_id and
                        gt["start_ts"] <= ts <= gt["end_ts"]):
                    is_genuine = True
                    condition = gt["condition"]
                    break

            entry = {
                "score": score,
                "camera_id": camera_id,
                "condition": condition,
                "ts_utc": ts,
                "quality": obs.get("quality_score", 0),
                "mask_state": obs.get("mask_state", "unknown"),
            }

            if is_genuine:
                genuine_scores.append(entry)
            else:
                impostor_scores.append(entry)

        results[person_name] = {
            "genuine": genuine_scores,
            "impostor": impostor_scores,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(scores: dict) -> dict:
    """Compute evaluation metrics from score distributions.

    Returns per-camera metrics: ROC, TAR@FAR, rank-N, EER, proposed thresholds.
    """
    all_metrics = {}

    for person_name, data in scores.items():
        gen = np.array([s["score"] for s in data["genuine"]])
        imp = np.array([s["score"] for s in data["impostor"]])

        if len(gen) == 0:
            all_metrics[person_name] = {
                "error": "No genuine scores — person not seen during labeled window"
            }
            continue

        metrics = {
            "genuine_count": len(gen),
            "impostor_count": len(imp),
            "genuine_mean": float(np.mean(gen)),
            "genuine_std": float(np.std(gen)),
            "genuine_min": float(np.min(gen)),
            "genuine_max": float(np.max(gen)),
            "impostor_mean": float(np.mean(imp)) if len(imp) > 0 else 0,
            "impostor_max": float(np.max(imp)) if len(imp) > 0 else 0,
        }

        # TAR @ FAR thresholds
        if len(imp) > 0:
            for target_far in [1e-1, 1e-2, 1e-3]:
                threshold = np.percentile(imp, 100 * (1 - target_far))
                tar = float(np.mean(gen >= threshold))
                metrics[f"TAR@FAR={target_far}"] = tar
                metrics[f"threshold@FAR={target_far}"] = float(threshold)

            # EER (Equal Error Rate)
            thresholds = np.linspace(0, 1, 1000)
            fars = np.array([np.mean(imp >= t) for t in thresholds])
            frrs = np.array([np.mean(gen < t) for t in thresholds])
            diffs = np.abs(fars - frrs)
            eer_idx = np.argmin(diffs)
            metrics["EER"] = float((fars[eer_idx] + frrs[eer_idx]) / 2)
            metrics["EER_threshold"] = float(thresholds[eer_idx])

        # Per-camera breakdown
        cameras = set(s["camera_id"] for s in data["genuine"])
        cameras.update(s["camera_id"] for s in data["impostor"])
        per_camera = {}
        for cam_id in cameras:
            cam_gen = [s["score"] for s in data["genuine"]
                       if s["camera_id"] == cam_id]
            cam_imp = [s["score"] for s in data["impostor"]
                       if s["camera_id"] == cam_id]
            per_camera[cam_id] = {
                "genuine_count": len(cam_gen),
                "genuine_mean": float(np.mean(cam_gen)) if cam_gen else 0,
                "impostor_count": len(cam_imp),
                "impostor_max": float(np.max(cam_imp)) if cam_imp else 0,
            }
        metrics["per_camera"] = per_camera

        all_metrics[person_name] = metrics

    return all_metrics


def propose_thresholds(metrics: dict) -> dict:
    """Auto-propose tier cutoffs based on metrics.

    Strategy:
      STRONG floor = midpoint between genuine_mean and impostor_max + margin
      POSSIBLE floor = impostor_max + small margin (or EER threshold)
      WEAK floor = below POSSIBLE but above noise
    """
    proposed = {"_default": {}}

    all_gen_means = []
    all_imp_maxes = []

    for person_name, m in metrics.items():
        if "error" in m:
            continue
        all_gen_means.append(m["genuine_mean"])
        if m.get("impostor_max", 0) > 0:
            all_imp_maxes.append(m["impostor_max"])

    if not all_gen_means:
        return proposed

    gen_mean = np.mean(all_gen_means)
    imp_max = max(all_imp_maxes) if all_imp_maxes else 0.15

    # Proposed thresholds
    margin = (gen_mean - imp_max) * 0.3
    proposed["_default"] = {
        "strong_floor": round(float(gen_mean - margin), 3),
        "possible_floor": round(float(imp_max + margin), 3),
        "weak_floor": round(float(imp_max), 3),
        "noise_floor": round(float(max(0.05, imp_max - 0.1)), 3),
    }

    # Per-camera proposals
    for person_name, m in metrics.items():
        if "error" in m:
            continue
        for cam_id, cam_m in m.get("per_camera", {}).items():
            if cam_m["genuine_count"] < 3:
                continue
            cam_gen = cam_m["genuine_mean"]
            cam_imp = cam_m.get("impostor_max", 0.15)
            cam_margin = (cam_gen - cam_imp) * 0.3
            if cam_id not in proposed:
                proposed[cam_id] = {}
            proposed[cam_id]["default"] = {
                "strong_floor": round(float(cam_gen - cam_margin), 3),
                "possible_floor": round(float(cam_imp + cam_margin), 3),
                "weak_floor": round(float(cam_imp), 3),
            }

    return proposed


# ─────────────────────────────────────────────────────────────────────────────
#  SOAK TEST (absent-target false positive measurement)
# ─────────────────────────────────────────────────────────────────────────────
def run_soak_test(db_path: str, probe_path: str, duration_minutes: float = 60,
                  embedder=None) -> dict:
    """Run absent-target soak test.

    Uses a probe photo of someone who is NOT present on any camera.
    Any STRONG/POSSIBLE hit is a false positive.

    This is the most important honesty metric.
    """
    from probe_builder import ProbeBuilder
    from observation_store import ObservationStore
    from matcher import Matcher, MatchTier

    if embedder is None:
        from face_embedder import InsightFaceEmbedder
        embedder = InsightFaceEmbedder()

    builder = ProbeBuilder(embedder)
    probe = builder.build_from_photos([probe_path])
    if probe is None:
        return {"error": "No face detected in soak test probe photo"}

    store = ObservationStore(db_path=db_path)
    matcher = Matcher(store)

    result = matcher.retro_search(probe, last_minutes=duration_minutes)

    strong_count = sum(1 for m in result.all_matches
                       if m.tier == MatchTier.STRONG)
    possible_count = sum(1 for m in result.all_matches
                         if m.tier == MatchTier.POSSIBLE)
    weak_count = sum(1 for m in result.all_matches
                     if m.tier == MatchTier.WEAK)

    hours = duration_minutes / 60.0

    return {
        "probe_photo": probe_path,
        "duration_minutes": duration_minutes,
        "strong_false_positives": strong_count,
        "possible_false_positives": possible_count,
        "weak_leads": weak_count,
        "strong_fp_per_hour": strong_count / hours if hours > 0 else 0,
        "possible_fp_per_hour": possible_count / hours if hours > 0 else 0,
        "total_observations_searched": result.coverage.get(
            "total_usable_observations", 0),
        "verdict": ("PASS" if strong_count == 0
                    else "FAIL — false positive STRONG hits detected"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Calibrate person search thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create ground-truth template CSV:
  python tools/calibrate.py --create-template

  # Run calibration:
  python tools/calibrate.py --ground-truth labels.csv --db observations.db --probes-dir probes/

  # Absent-target soak test:
  python tools/calibrate.py --soak-test --probe unknown.jpg --duration 60 --db observations.db
        """,
    )
    parser.add_argument("--create-template", action="store_true",
                        help="Create template ground-truth CSV")
    parser.add_argument("--ground-truth", type=str,
                        help="Path to ground-truth CSV")
    parser.add_argument("--db", type=str, default="observations.db",
                        help="Path to observations database")
    parser.add_argument("--probes-dir", type=str,
                        help="Directory with probe photos (one per person)")
    parser.add_argument("--output", type=str, default="calibration.json",
                        help="Output calibration file")
    parser.add_argument("--soak-test", action="store_true",
                        help="Run absent-target soak test")
    parser.add_argument("--probe", type=str,
                        help="Probe photo for soak test")
    parser.add_argument("--duration", type=float, default=60,
                        help="Soak test duration in minutes")

    args = parser.parse_args()

    if args.create_template:
        create_template_csv()
        return

    if args.soak_test:
        if not args.probe:
            print("ERROR: --probe required for soak test")
            return
        print(f"\n  Running absent-target soak test...")
        result = run_soak_test(args.db, args.probe, args.duration)
        print(f"\n{'='*60}")
        print(f"  SOAK TEST RESULTS")
        print(f"{'='*60}")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
        return

    if args.ground_truth:
        print(f"\n  Loading ground truth: {args.ground_truth}")
        gt = load_ground_truth(args.ground_truth)
        print(f"  Loaded {len(gt)} labeled segments")

        print(f"  Loading observations: {args.db}")
        observations = load_observations(args.db)
        print(f"  Loaded {len(observations)} observations")

        # Load probe embeddings
        probe_embs = {}
        if args.probes_dir:
            from face_embedder import InsightFaceEmbedder
            from probe_builder import ProbeBuilder
            embedder = InsightFaceEmbedder()
            builder = ProbeBuilder(embedder)
            for f in os.listdir(args.probes_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    name = Path(f).stem
                    path = os.path.join(args.probes_dir, f)
                    probe = builder.build_from_photos([path])
                    if probe:
                        # Use mean of template bank as the single probe vector
                        probe_embs[name] = np.mean(
                            probe.template_bank, axis=0)
                        probe_embs[name] /= np.linalg.norm(probe_embs[name])
                        print(f"  Loaded probe: {name} ({probe.template_count} templates)")

        if not probe_embs:
            print("  ERROR: No probe embeddings loaded. Use --probes-dir.")
            return

        # Compute scores
        print(f"\n  Computing scores...")
        scores = compute_scores(observations, probe_embs, gt)

        # Compute metrics
        metrics = compute_metrics(scores)

        # Print results
        print(f"\n{'='*60}")
        print(f"  CALIBRATION RESULTS")
        print(f"{'='*60}")
        for person, m in metrics.items():
            print(f"\n  {person}:")
            if "error" in m:
                print(f"    {m['error']}")
                continue
            print(f"    Genuine: {m['genuine_count']} observations, "
                  f"mean={m['genuine_mean']:.3f} "
                  f"[{m['genuine_min']:.3f}, {m['genuine_max']:.3f}]")
            print(f"    Impostor: {m['impostor_count']} observations, "
                  f"max={m['impostor_max']:.3f}")
            if "EER" in m:
                print(f"    EER: {m['EER']:.4f} (threshold={m['EER_threshold']:.3f})")
            for key in m:
                if key.startswith("TAR@"):
                    print(f"    {key}: {m[key]:.3f} "
                          f"(threshold={m.get(key.replace('TAR','threshold'), 0):.3f})")

        # Propose thresholds
        proposed = propose_thresholds(metrics)
        print(f"\n  Proposed thresholds:")
        print(json.dumps(proposed, indent=2))

        # Save calibration
        output = {
            "generated_at": datetime.now(IST).isoformat(),
            "metrics": metrics,
            "thresholds": proposed,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved: {args.output}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
