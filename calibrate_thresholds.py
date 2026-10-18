"""
calibrate_thresholds.py
=======================
Per-modality threshold calibration for the gate monitor face recognition pipeline.

Computes intra-person and inter-person similarity distributions for both full-face
and periocular embeddings from the existing enrolled gallery, then suggests thresholds
that maximise the separation between genuine (same person) and impostor (different
person) scores.

HARDENED (open-set fix):
  - Minimum gallery size check: refuses to emit periocular thresholds with <5 people
  - Full-face calibrated threshold is clamped to never go below the absolute floor
    (GATE_ABS_FLOOR_FULLFACE) regardless of what the tiny-gallery split suggests
  - Reports "margin over runner-up" per calibration sample (the gap between best
    score and the second-best/nearest-impostor score) — thin margins are a warning
    sign that the threshold is riding too close to the impostor distribution

Usage:
    .\.python\python.exe calibrate_thresholds.py

No camera required — reads the known_faces/ gallery that is already on disk.

OUTPUT:
    - Score distribution summary (min, mean, max, std) for each modality
    - Suggested threshold for GATE_FACE_THRESHOLD and GATE_PERIOCULAR_THRESHOLD
    - Explicit warning if fewer than 5 people are enrolled
    - Margin-over-runner-up statistics per calibration sample
    - Instructions for applying the suggested values in .env

NOTE: "Suggested" means the midpoint between the worst intra-person score and the
best inter-person score. With only 2 enrolled people this has very low statistical
power — the script will say so clearly. Do NOT treat the output as ground truth;
run a live walk-through with the new thresholds and adjust based on false accepts
and false rejects observed in practice.
"""

import sys
import os
import json
from pathlib import Path
from itertools import combinations

import numpy as np

# Force UTF-8 output on Windows (cp1252 can't encode ⚠, ✓, ✗ etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity between two L2-normalised vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a / na, b / nb))


def _nac(score: float, nearest: float) -> float:
    """Neighbourhood-Aware Cosine — same formula as face_id.py."""
    return score - 0.5 * nearest


def _stats(values: list) -> dict:
    if not values:
        return {"n": 0, "min": None, "mean": None, "max": None, "std": None}
    a = np.array(values, dtype=float)
    return {
        "n":    len(a),
        "min":  float(np.min(a)),
        "mean": float(np.mean(a)),
        "max":  float(np.max(a)),
        "std":  float(np.std(a)),
    }


def _suggest_threshold(intra: list, inter: list) -> tuple:
    """
    Suggest a threshold that sits between the worst genuine score and the
    best impostor score.

    Returns (suggested_threshold, gap_size, is_separable).
    gap_size < 0 means the distributions overlap — no clean threshold exists.
    """
    if not intra or not inter:
        return None, None, False
    worst_genuine = float(np.min(intra))   # lowest same-person score
    best_impostor = float(np.max(inter))   # highest different-person score
    gap = worst_genuine - best_impostor
    if gap > 0:
        midpoint = best_impostor + gap / 2.0
        return round(midpoint, 3), round(gap, 3), True
    else:
        # Distributions overlap — suggest a compromise that minimises combined error
        all_scores = [(s, "genuine") for s in intra] + [(s, "impostor") for s in inter]
        all_scores.sort(key=lambda x: x[0])
        best_thresh = None
        best_err = len(all_scores) + 1
        for i in range(len(all_scores) - 1):
            t = (all_scores[i][0] + all_scores[i + 1][0]) / 2.0
            fa = sum(1 for s, label in all_scores if label == "impostor" and s >= t)
            fr = sum(1 for s, label in all_scores if label == "genuine" and s < t)
            err = fa + fr
            if err < best_err:
                best_err = err
                best_thresh = t
        return (round(best_thresh, 3) if best_thresh is not None else None,
                round(gap, 3), False)


def _compute_margin_stats(per_person_embs: dict, gallery: dict,
                          modality: str = "full") -> list:
    """
    Compute margin-over-runner-up for every calibration sample.

    For each photo embedding, compute similarity against all gallery templates
    and report the gap between the best (correct) match and the nearest impostor.

    Returns a list of dicts with margin info.
    """
    margins = []
    names = list(gallery.keys())

    for true_name, embs in per_person_embs.items():
        key = modality  # "full" or "periocular"
        true_template = gallery[true_name].get(key)
        if true_template is None:
            continue

        for probe in embs:
            scores = {}
            for name in names:
                template = gallery[name].get(key)
                if template is None:
                    continue
                scores[name] = _cosine(probe, template)

            if not scores:
                continue

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_name, best_score = sorted_scores[0]
            second_name, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("N/A", 0.0)

            is_correct = (best_name == true_name)
            margin = best_score - second_score

            margins.append({
                "true_name": true_name,
                "matched_name": best_name,
                "correct": is_correct,
                "best_score": round(best_score, 4),
                "runner_up_name": second_name,
                "runner_up_score": round(second_score, 4),
                "margin": round(margin, 4),
            })

    return margins


def run_calibration(known_faces_dir: str = "known_faces"):
    """Main calibration routine."""
    from config import GATE_ABS_FLOOR_FULLFACE, GATE_ABS_FLOOR_PERIOCULAR

    print(f"\n{'='*60}")
    print(f"  THRESHOLD CALIBRATION TOOL")
    print(f"{'='*60}\n")

    # ── 1. Load InsightFace ───────────────────────────────────────────────────
    print("Loading InsightFace models...")
    from face_id import build_known_encodings, _load_insightface, _get_face_and_embedding, _get_embedding
    import cv2

    if not _load_insightface():
        print("ERROR: Could not load InsightFace. Check installation.")
        return

    # ── 2. Load gallery ───────────────────────────────────────────────────────
    gallery = build_known_encodings(known_faces_dir)
    if not gallery:
        print("ERROR: No enrolled people found in known_faces/")
        return

    n_people = len(gallery)
    print(f"\nEnrolled people: {n_people}")
    for name, entry in gallery.items():
        bc = entry.get("bucket_counts", {})
        bc_str = ", ".join(f"{k}={v}" for k, v in bc.items()) if bc else "no bucket metadata"
        print(f"  {name}: {bc_str}")

    # ── 3. Statistical power warning ─────────────────────────────────────────
    print()
    if n_people < 2:
        print("⚠  CALIBRATION SKIPPED: Only 1 person enrolled.")
        print("   Inter-person similarity cannot be computed with 1 person.")
        print("   Enroll at least 2 people for meaningful calibration.")
        return
    if n_people < 5:
        print(f"⚠  INSUFFICIENT GALLERY DIVERSITY: Only {n_people} people enrolled.")
        print(f"   Open-set rejection thresholds require ≥5 distinct people to calibrate reliably.")
        if n_people < 3:
            print("   Inter-person similarity is based on a single person pair.")
            print("   Suggested thresholds below have very low reliability.")
        print("   Treat the output as a starting point only.")
    else:
        print(f"✓  {n_people} people enrolled — inter-person statistics are meaningful.")

    # ── 4. Per-photo embeddings for held-out evaluation ───────────────────────
    # Build per-photo embeddings (individual, not averaged gallery templates)
    # so we can compute held-out genuine scores (photo vs. gallery average)
    print("\nBuilding per-photo embeddings for calibration...")

    # Buckets whose photos contribute to full-face template
    _FULL_FACE_BUCKETS = {"frontal_unmasked", "angled_unmasked"}

    per_person_full = {}   # name -> list of individual full embeddings
    per_person_peri = {}   # name -> list of individual periocular embeddings

    base = Path(known_faces_dir)
    for person_dir in sorted(base.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith("_"):
            continue
        name = person_dir.name
        if name not in gallery:
            continue

        meta_path = person_dir / "bucket_metadata.json"
        filename_to_bucket = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                for fi in meta.get("frames", []):
                    filename_to_bucket[fi["filename"]] = fi["bucket"]
            except Exception:
                pass

        full_embs = []
        peri_embs = []

        photos = (list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) +
                  list(person_dir.glob("*.png")) + list(person_dir.glob("*.webp")))

        for photo_path in photos:
            fname = photo_path.name
            bucket = filename_to_bucket.get(fname)

            # Determine usage
            if filename_to_bucket:
                if bucket is None:
                    continue
                use_full = bucket in _FULL_FACE_BUCKETS
                use_peri = True  # All buckets contribute to periocular
            else:
                use_full = True
                use_peri = True

            img_bgr = cv2.imread(str(photo_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            try:
                face_obj, emb, _ = _get_face_and_embedding(img_rgb)
                
                # Fallback to raw frame if crop fails detection
                if emb is None:
                    raw_path = photo_path.parent / "_raw" / photo_path.name
                    if raw_path.exists():
                        import cv2
                        raw_bgr = cv2.imread(str(raw_path))
                        if raw_bgr is not None:
                            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                            face_obj, emb, _ = _get_face_and_embedding(raw_rgb)
                
                if emb is not None and use_full:
                    full_embs.append(emb)
                if face_obj is not None and use_peri:
                    peri_emb, _ = _get_embedding(img_rgb, periocular=True, face_obj=face_obj)
                    if peri_emb is None:
                        # Fallback for periocular on raw frame
                        raw_path = photo_path.parent / "_raw" / photo_path.name
                        if raw_path.exists():
                            import cv2
                            raw_bgr = cv2.imread(str(raw_path))
                            if raw_bgr is not None:
                                raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                                peri_emb, _ = _get_embedding(raw_rgb, periocular=True, face_obj=face_obj)
                    if peri_emb is not None:
                        peri_embs.append(peri_emb)
            except Exception as e:
                print(f"  Warning: {name}/{fname} failed: {e}")

        per_person_full[name] = full_embs
        per_person_peri[name] = peri_embs
        print(f"  {name}: {len(full_embs)} full-face photos, {len(peri_embs)} periocular photos")

    # ── 5. Compute intra-person scores (genuine) ─────────────────────────────
    # Leave-one-out: for each photo, compute similarity against the gallery
    # template built from ALL OTHER photos of the same person.

    print("\nComputing genuine (same-person) similarity scores...")

    full_intra = []
    peri_intra = []

    for name, embs in per_person_full.items():
        gallery_full = gallery[name]["full"]
        if len(embs) < 2:
            # With only 1 photo, use it against the gallery (no hold-out possible)
            s = _cosine(embs[0], gallery_full) if embs else None
            if s is not None:
                full_intra.append(s)
        else:
            for i, probe in enumerate(embs):
                # Leave-one-out: gallery from all other photos
                others = [e for j, e in enumerate(embs) if j != i]
                loo_avg = np.mean(others, axis=0)
                loo_avg = loo_avg / np.linalg.norm(loo_avg)
                full_intra.append(_cosine(probe, loo_avg))

    for name, embs in per_person_peri.items():
        gallery_peri = gallery[name].get("periocular")
        if gallery_peri is None:
            continue
        if len(embs) < 2:
            s = _cosine(embs[0], gallery_peri) if embs else None
            if s is not None:
                peri_intra.append(s)
        else:
            for i, probe in enumerate(embs):
                others = [e for j, e in enumerate(embs) if j != i]
                loo_avg = np.mean(others, axis=0)
                loo_avg = loo_avg / np.linalg.norm(loo_avg)
                peri_intra.append(_cosine(probe, loo_avg))

    # ── 6. Compute inter-person scores (impostor) ─────────────────────────────
    print("Computing impostor (different-person) similarity scores...")

    full_inter = []
    peri_inter = []

    names = list(gallery.keys())
    for name_a, name_b in combinations(names, 2):
        # Full-face: every photo of A vs gallery template of B (and vice versa)
        gal_b_full = gallery[name_b]["full"]
        gal_a_full = gallery[name_a]["full"]
        for emb in per_person_full.get(name_a, []):
            full_inter.append(_cosine(emb, gal_b_full))
        for emb in per_person_full.get(name_b, []):
            full_inter.append(_cosine(emb, gal_a_full))

        # Periocular
        gal_b_peri = gallery[name_b].get("periocular")
        gal_a_peri = gallery[name_a].get("periocular")
        if gal_b_peri is not None and gal_a_peri is not None:
            for emb in per_person_peri.get(name_a, []):
                peri_inter.append(_cosine(emb, gal_b_peri))
            for emb in per_person_peri.get(name_b, []):
                peri_inter.append(_cosine(emb, gal_a_peri))

    # ── 7. Report ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FULL-FACE EMBEDDING SCORES  (raw cosine, not NAC)")
    print(f"{'='*60}")
    si = _stats(full_intra)
    sx = _stats(full_inter)
    if si['n'] > 0:
        print(f"  Genuine (same person) : n={si['n']:3d}  "
              f"min={si['min']:.3f}  mean={si['mean']:.3f}  "
              f"max={si['max']:.3f}  std={si['std']:.3f}")
    else:
        print(f"  Genuine (same person) : n=0  (no full-face embeddings)")
    if sx['n'] > 0:
        print(f"  Impostor (diff person): n={sx['n']:3d}  "
              f"min={sx['min']:.3f}  mean={sx['mean']:.3f}  "
              f"max={sx['max']:.3f}  std={sx['std']:.3f}")
    else:
        print(f"  Impostor (diff person): n=0")

    full_thresh_nac = None
    full_thresh = None
    if n_people < 5:
        print(f"\n  ⚠ FULL-FACE THRESHOLD REFUSED: insufficient gallery diversity "
              f"({n_people} < 5 people)")
        print(f"  Cannot calibrate full-face open-set rejection with this gallery size.")
        print(f"  Using safe default margin (GATE_FACE_VOTE_FLOOR = 0.15) instead.")
        print(f"  Enroll ≥5 distinct people to enable full-face threshold calibration.")
    else:
        full_thresh, full_gap, full_sep = _suggest_threshold(full_intra, full_inter)
        if full_thresh is not None:
            sep_note = "✓ clean gap" if full_sep else f"⚠ distributions OVERLAP by {abs(full_gap):.3f}"
            print(f"\n  Suggested raw calibrated threshold = {full_thresh}  ({sep_note})")
            if not full_sep:
                print(f"  (Distributions overlap — any threshold will have both false accepts and rejects)")
        else:
            print(f"\n  Could not compute threshold (insufficient data)")

    print(f"\n{'='*60}")
    print(f"  PERIOCULAR EMBEDDING SCORES  (raw cosine, not NAC)")
    print(f"{'='*60}")
    pi = _stats(peri_intra)
    px = _stats(peri_inter)
    if pi['n'] > 0:
        print(f"  Genuine (same person) : n={pi['n']:3d}  "
              f"min={pi['min']:.3f}  mean={pi['mean']:.3f}  "
              f"max={pi['max']:.3f}  std={pi['std']:.3f}")
    else:
        print(f"  Genuine (same person) : n=0  (no periocular embeddings built)")
    if px['n'] > 0:
        print(f"  Impostor (diff person): n={px['n']:3d}  "
              f"min={px['min']:.3f}  mean={px['mean']:.3f}  "
              f"max={px['max']:.3f}  std={px['std']:.3f}")
    else:
        print(f"  Impostor (diff person): n=0  (need ≥2 people with periocular embeddings)")

    # HARDENING: refuse to emit periocular threshold with <5 people
    peri_thresh = None
    peri_thresh_nac = None
    if n_people < 5:
        print(f"\n  ⚠ PERIOCULAR THRESHOLD REFUSED: insufficient gallery diversity "
              f"({n_people} < 5 people)")
        print(f"  Cannot calibrate periocular open-set rejection with this gallery size.")
        print(f"  Using safe default floor (GATE_ABS_FLOOR_PERIOCULAR = "
              f"{GATE_ABS_FLOOR_PERIOCULAR}) instead.")
        print(f"  Enroll ≥5 distinct people to enable periocular threshold calibration.")
    else:
        peri_thresh, peri_gap, peri_sep = _suggest_threshold(peri_intra, peri_inter)
        if peri_thresh is not None:
            sep_note = "✓ clean gap" if peri_sep else f"⚠ distributions OVERLAP by {abs(peri_gap):.3f}"
            # Clamp periocular threshold too
            if peri_thresh < GATE_ABS_FLOOR_PERIOCULAR:
                print(f"\n  ⚠ Raw calibrated periocular threshold ({peri_thresh}) is BELOW "
                      f"absolute floor ({GATE_ABS_FLOOR_PERIOCULAR})")
                print(f"  Clamping to absolute floor.")
                peri_thresh = GATE_ABS_FLOOR_PERIOCULAR
            print(f"\n  Suggested GATE_PERIOCULAR_THRESHOLD = {peri_thresh}  ({sep_note})")
            if not peri_sep:
                print(f"  (Periocular overlap is expected — periocular accuracy is inherently lower than full-face)")
        else:
            print(f"\n  Could not compute periocular threshold (insufficient data)")

    # ── 8. Margin-over-runner-up statistics ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MARGIN-OVER-RUNNER-UP STATISTICS")
    print(f"{'='*60}")
    print(f"  (Gap between best match and nearest impostor per sample)")
    print(f"  Thin margins (<0.05) indicate risk of misidentification.\n")

    full_margins = _compute_margin_stats(per_person_full, gallery, "full")
    if full_margins:
        print(f"  FULL-FACE ({len(full_margins)} samples):")
        thin_count = 0
        for m in full_margins:
            tag = "✓" if m["correct"] else "✗ WRONG"
            margin_tag = " ⚠THIN" if m["margin"] < 0.05 else ""
            print(f"    {m['true_name']:15s} → matched={m['matched_name']:15s} "
                  f"score={m['best_score']:.3f}  runner-up={m['runner_up_name']}={m['runner_up_score']:.3f}  "
                  f"margin={m['margin']:.3f} {tag}{margin_tag}")
            if m["margin"] < 0.05:
                thin_count += 1
        if thin_count > 0:
            print(f"\n  ⚠ {thin_count}/{len(full_margins)} samples have THIN margins (<0.05)")
            print(f"    This means the threshold is riding close to the impostor distribution.")
        else:
            print(f"\n  ✓ All margins are ≥0.05 — reasonable separation.")

    peri_margins = _compute_margin_stats(per_person_peri, gallery, "periocular")
    if peri_margins:
        print(f"\n  PERIOCULAR ({len(peri_margins)} samples):")
        thin_count = 0
        for m in peri_margins:
            tag = "✓" if m["correct"] else "✗ WRONG"
            margin_tag = " ⚠THIN" if m["margin"] < 0.05 else ""
            print(f"    {m['true_name']:15s} → matched={m['matched_name']:15s} "
                  f"score={m['best_score']:.3f}  runner-up={m['runner_up_name']}={m['runner_up_score']:.3f}  "
                  f"margin={m['margin']:.3f} {tag}{margin_tag}")
            if m["margin"] < 0.05:
                thin_count += 1
        if thin_count > 0:
            print(f"\n  ⚠ {thin_count}/{len(peri_margins)} periocular samples have THIN margins")

    # ── 9. NAC-adjusted thresholds ────────────────────────────────────────────
    # The pipeline uses NAC = score - 0.5 * nearest_other_score.
    # Simulate what NAC does to the thresholds.
    full_thresh_nac = None
    if full_inter and full_intra and full_thresh is not None and n_people >= 5:
        avg_impostor = float(np.mean(full_inter))
        full_thresh_nac  = round(full_thresh  - avg_impostor, 3)

    if peri_inter and peri_intra and peri_thresh is not None:
        avg_impostor_peri = float(np.mean(peri_inter))
        peri_thresh_nac  = round(peri_thresh  - avg_impostor_peri, 3)

    if full_thresh_nac is not None or peri_thresh_nac is not None:
        print(f"\n{'='*60}")
        print(f"  NAC-ADJUSTED THRESHOLDS  (what face_id.py actually compares)")
        print(f"  (NAC = raw_score - nearest_impostor_score)")
        print(f"{'='*60}")
        if full_thresh_nac is not None:
            print(f"  GATE_FACE_THRESHOLD        = {full_thresh_nac}")
        if peri_thresh_nac is not None:
            print(f"  GATE_PERIOCULAR_THRESHOLD  = {peri_thresh_nac}")

    # ── 10. How to apply ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  HOW TO APPLY")
    print(f"{'='*60}")
    print(f"  Add these lines to your .env file:")
    print()
    if full_thresh_nac is not None:
        print(f"  GATE_FACE_THRESHOLD={full_thresh_nac}")
    elif full_thresh is not None:
        print(f"  GATE_FACE_THRESHOLD={full_thresh}")
    if peri_thresh_nac is not None:
        print(f"  GATE_PERIOCULAR_THRESHOLD={peri_thresh_nac}")
    elif peri_thresh is not None:
        print(f"  GATE_PERIOCULAR_THRESHOLD={peri_thresh}")
    else:
        print(f"  # GATE_PERIOCULAR_THRESHOLD not calibrated (need ≥5 people)")
        print(f"  # Using absolute floor: GATE_ABS_FLOOR_PERIOCULAR={GATE_ABS_FLOOR_PERIOCULAR}")
    print()
    print(f"  Absolute similarity floors (independent of calibration):")
    print(f"  GATE_ABS_FLOOR_FULLFACE={GATE_ABS_FLOOR_FULLFACE}")
    print(f"  GATE_ABS_FLOOR_PERIOCULAR={GATE_ABS_FLOOR_PERIOCULAR}")
    print()
    print(f"  Then re-run the pipeline. Do a manual walk-through (masked + unmasked)")
    print(f"  and check if 'needs_review' drops from 1 to 0 in gate_events.db.")
    print(f"\n  ⚠  These thresholds are SUGGESTIONS based on enrollment data only.")
    print(f"     Validate on live footage before deploying in production.")
    if n_people < 5:
        print(f"\n  ⚠  LOW RELIABILITY: Only {n_people} person(s) enrolled.")
        print(f"     Enroll ≥5 people for statistically meaningful thresholds.")
        print(f"     Periocular threshold calibration requires ≥5 people.")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate face recognition thresholds.")
    parser.add_argument("--known-faces", type=str, default="known_faces",
                        help="Path to known_faces directory (default: known_faces)")
    args = parser.parse_args()
    run_calibration(args.known_faces)
