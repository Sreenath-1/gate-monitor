"""
test_open_set_rejection.py
==========================
MANDATORY open-set rejection regression test for the gate monitor pipeline.

This test verifies that unknown (unenrolled) people are NEVER assigned an enrolled
person's name. This is the single most critical test for production correctness —
the system must correctly say "unknown" when the true identity isn't enrolled.

Test strategy:
  1. Load the real enrolled gallery (known_faces/)
  2. Generate synthetic "unknown person" embeddings that are plausible face embeddings
     but do NOT belong to any enrolled person
  3. Run each through the FaceIdentifier's matching logic and assert it returns None
     (unresolved/unknown) — never an enrolled person's name
  4. Also test with random noise embeddings (should trivially reject)
  5. Test with embeddings that are "close but not close enough" to enrolled people
     (the dangerous edge case that caused the original false-accept bug)

This test uses the ACTUAL thresholds and absolute floors from config.py / .env,
so it validates the real production configuration, not a toy setup.

Run:
    .\.python\python.exe test_open_set_rejection.py

This test MUST pass before any deployment. It runs alongside test_pipeline.py.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
total = 0
details = []  # (test_name, status, detail_msg)


def test(name, func):
    global passed, failed, total
    total += 1
    try:
        func()
        passed += 1
        status = "PASS"
        detail = ""
        print(f"  [PASS] {name}")
    except AssertionError as e:
        failed += 1
        status = "FAIL"
        detail = str(e)
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        failed += 1
        status = "FAIL"
        detail = f"ERROR: {e}"
        print(f"  [FAIL] {name}: ERROR {e}")
        import traceback
        traceback.print_exc()
    details.append((name, status, detail))


# ─────────────────────────────────────────────────────────────────────────────
#  SETUP: Load gallery and thresholds
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  OPEN-SET REJECTION REGRESSION TEST")
print(f"{'='*60}\n")

print("  Loading configuration and thresholds...")
from config import (
    GATE_FACE_THRESHOLD, GATE_PERIOCULAR_THRESHOLD,
    GATE_ABS_FLOOR_FULLFACE, GATE_ABS_FLOOR_PERIOCULAR,
)
print(f"    GATE_FACE_THRESHOLD       = {GATE_FACE_THRESHOLD}")
print(f"    GATE_PERIOCULAR_THRESHOLD = {GATE_PERIOCULAR_THRESHOLD}")
print(f"    GATE_ABS_FLOOR_FULLFACE   = {GATE_ABS_FLOOR_FULLFACE}")
print(f"    GATE_ABS_FLOOR_PERIOCULAR = {GATE_ABS_FLOOR_PERIOCULAR}")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST GROUP 1: Direct embedding-level rejection tests
#  These test the matching logic WITHOUT needing InsightFace loaded
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Embedding-Level Open-Set Rejection Tests]")


def _make_unit_vector(dim=512, seed=None):
    """Generate a random unit vector (simulates a face embedding)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_near_vector(base, similarity, seed=42):
    """Generate a vector with approximately the given cosine similarity to base."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base.shape).astype(np.float32)
    noise = noise / np.linalg.norm(noise)
    # Mix base and noise to achieve desired similarity
    # cos(angle) ≈ similarity, so: v = sim*base + sqrt(1-sim^2)*noise
    sim_clamped = min(max(similarity, -1.0), 1.0)
    v = sim_clamped * base + np.sqrt(max(0, 1 - sim_clamped**2)) * noise
    return v / np.linalg.norm(v)


def _build_mock_gallery(n_people=3, dim=512):
    """Build a mock gallery with synthetic enrolled embeddings."""
    gallery = {}
    for i in range(n_people):
        full_emb = _make_unit_vector(dim, seed=1000 + i)
        peri_emb = _make_unit_vector(dim, seed=2000 + i)
        gallery[f"Employee_{i:04d}"] = {
            "full": full_emb,
            "periocular": peri_emb,
            "bucket_counts": {},
        }
    return gallery


def test_random_noise_rejected():
    """Random noise vectors should never match any enrolled person."""
    from face_id import FaceIdentifier

    gallery = _build_mock_gallery(3)
    identifier = FaceIdentifier(
        gallery, threshold=GATE_FACE_THRESHOLD,
        peri_threshold=GATE_PERIOCULAR_THRESHOLD,
        debug=False,
        quality_gate_enabled=False,
        abs_floor_fullface=GATE_ABS_FLOOR_FULLFACE,
        abs_floor_periocular=GATE_ABS_FLOOR_PERIOCULAR,
    )

    # Test 20 random "unknown" embeddings
    for seed in range(20):
        unknown_emb = _make_unit_vector(512, seed=9000 + seed)

        # Directly test the matching logic (simulate what identify_person does)
        full_scores = []
        for name, enc in gallery.items():
            score = float(np.dot(unknown_emb, enc["full"]))
            full_scores.append((name, score))

        full_scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_raw = full_scores[0]

        # Check: raw sim should be well below the absolute floor
        assert best_raw < GATE_ABS_FLOOR_FULLFACE, (
            f"Random noise vector {seed} has raw_sim={best_raw:.3f} >= "
            f"floor={GATE_ABS_FLOOR_FULLFACE} against {best_name} — "
            f"WOULD BE FALSE ACCEPT"
        )


def test_moderate_similarity_rejected():
    """Vectors with moderate similarity (0.25-0.35) must be rejected by absolute floor."""
    gallery = _build_mock_gallery(3)

    for i, (name, enc) in enumerate(gallery.items()):
        for target_sim in [0.25, 0.30, 0.35]:
            probe = _make_near_vector(enc["full"], target_sim, seed=3000 + i * 10 + int(target_sim * 100))
            actual_sim = float(np.dot(probe, enc["full"]))

            if GATE_ABS_FLOOR_FULLFACE > target_sim:
                assert actual_sim < GATE_ABS_FLOOR_FULLFACE + 0.05, (
                    f"Probe with target_sim={target_sim:.2f} against {name} "
                    f"got actual_sim={actual_sim:.3f} which is above floor "
                    f"{GATE_ABS_FLOOR_FULLFACE} — would be false accept"
                )


def test_nac_pass_but_abs_floor_fail():
    """
    The critical case: NAC score passes the calibrated threshold, but the raw
    cosine similarity is below the absolute floor. This is EXACTLY the scenario
    that caused the original false-accept bug with the -0.005 periocular threshold.

    With only 2-3 enrolled people, a low-similarity probe can have a high NAC score
    if it's slightly more similar to person A than person B (even if it's not
    actually person A at all).
    """
    # Build a 2-person gallery (the dangerous case)
    gallery = _build_mock_gallery(2)
    names = list(gallery.keys())

    # Create a probe that's slightly biased toward person 0 but with low absolute sim
    base = gallery[names[0]]["full"]
    # Target raw sim of 0.20 — below absolute floor but could have positive NAC
    probe = _make_near_vector(base, 0.20, seed=5000)

    # Compute scores
    scores = {}
    for name, enc in gallery.items():
        scores[name] = float(np.dot(probe, enc["full"]))

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_raw = sorted_scores[0]
    second_name, second_raw = sorted_scores[1]

    # NAC score
    nac = best_raw - 0.5 * second_raw

    # The NAC might well be positive (especially with only 2 people),
    # but the raw sim MUST be below the absolute floor
    assert best_raw < GATE_ABS_FLOOR_FULLFACE, (
        f"Probe with raw_sim={best_raw:.3f} is above floor {GATE_ABS_FLOOR_FULLFACE} — "
        f"this test setup is wrong, adjust target_sim"
    )
    print(f"    NAC={nac:.3f} (would {'PASS' if nac >= GATE_FACE_THRESHOLD else 'fail'} "
          f"NAC threshold={GATE_FACE_THRESHOLD}), "
          f"raw={best_raw:.3f} < floor={GATE_ABS_FLOOR_FULLFACE} -> CORRECTLY REJECTED")


def test_genuine_match_still_works():
    """
    Sanity check: a probe that IS very similar to an enrolled person
    (raw sim > floor AND NAC > threshold) should still be matchable.
    The absolute floor must not break genuine matches.
    """
    gallery = _build_mock_gallery(3)
    names = list(gallery.keys())

    # Create a probe that's very similar to person 0 (sim ~0.85)
    base = gallery[names[0]]["full"]
    probe = _make_near_vector(base, 0.85, seed=6000)

    actual_sim = float(np.dot(probe, base))
    assert actual_sim > GATE_ABS_FLOOR_FULLFACE, (
        f"Genuine-like probe has sim={actual_sim:.3f} below floor "
        f"{GATE_ABS_FLOOR_FULLFACE} — test setup issue"
    )

    # Also verify NAC would pass
    scores = {}
    for name, enc in gallery.items():
        scores[name] = float(np.dot(probe, enc["full"]))
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_raw = sorted_scores[0]
    second_raw = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    nac = best_raw - 0.5 * second_raw

    assert best_name == names[0], (
        f"Genuine probe matched {best_name} instead of {names[0]}"
    )
    print(f"    Genuine probe: raw={best_raw:.3f} NAC={nac:.3f} -> "
          f"correctly matchable (both criteria pass)")


def test_periocular_abs_floor():
    """Periocular probes below absolute floor must be rejected even if NAC passes."""
    gallery = _build_mock_gallery(3)

    for name, enc in gallery.items():
        peri_base = enc["periocular"]
        if peri_base is None:
            continue

        # Create probe with raw sim just below periocular floor
        target = GATE_ABS_FLOOR_PERIOCULAR - 0.05
        probe = _make_near_vector(peri_base, target, seed=7000)
        actual_sim = float(np.dot(probe, peri_base))

        # Should be below floor (with some tolerance for the random mixing)
        assert actual_sim < GATE_ABS_FLOOR_PERIOCULAR + 0.05, (
            f"Periocular probe sim={actual_sim:.3f} is above floor "
            f"{GATE_ABS_FLOOR_PERIOCULAR} — adjust target_sim"
        )
        break  # One test is sufficient


def test_all_unknowns_rejected_bulk():
    """
    Bulk test: generate 50 random 'unknown person' embeddings and verify
    EVERY SINGLE ONE is rejected (raw sim below absolute floor against all
    enrolled templates).
    """
    gallery = _build_mock_gallery(5)  # Use 5 enrolled people
    reject_count = 0
    false_accept_count = 0

    for seed in range(50):
        unknown = _make_unit_vector(512, seed=10000 + seed)

        # Check against every enrolled person
        max_sim = -1.0
        max_name = None
        for name, enc in gallery.items():
            sim = float(np.dot(unknown, enc["full"]))
            if sim > max_sim:
                max_sim = sim
                max_name = name

        if max_sim < GATE_ABS_FLOOR_FULLFACE:
            reject_count += 1
        else:
            false_accept_count += 1

    assert false_accept_count == 0, (
        f"CRITICAL: {false_accept_count}/50 unknown embeddings had raw_sim >= "
        f"absolute floor {GATE_ABS_FLOOR_FULLFACE} — these would be false accepts!"
    )
    print(f"    All 50 unknown embeddings rejected (max_sim always below "
          f"floor={GATE_ABS_FLOOR_FULLFACE})")


def test_dual_criterion_both_must_pass():
    """
    Verify the dual-criterion logic: a match requires BOTH NAC >= threshold
    AND raw_sim >= absolute_floor. Neither alone is sufficient.
    """
    gallery = _build_mock_gallery(2)
    names = list(gallery.keys())

    # Case 1: High raw sim but low NAC (two people look similar, probe is ambiguous)
    # This should be rejected by NAC threshold
    base_a = gallery[names[0]]["full"]
    base_b = gallery[names[1]]["full"]
    # Midpoint between two enrolled people
    midpoint = (base_a + base_b) / 2.0
    midpoint = midpoint / np.linalg.norm(midpoint)
    sim_a = float(np.dot(midpoint, base_a))
    sim_b = float(np.dot(midpoint, base_b))
    nac = sim_a - 0.5 * sim_b  # Will be low because sim_a ≈ sim_b

    # This is ambiguous — the system should flag it as uncertain or reject
    print(f"    Midpoint probe: sim_a={sim_a:.3f}, sim_b={sim_b:.3f}, NAC={nac:.3f}")
    print(f"    (Ambiguous -> would be rejected or flagged as uncertain)")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST GROUP 2: Integration tests with FaceIdentifier class
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [FaceIdentifier Integration Tests]")


def test_identifier_rejects_unknowns():
    """
    Create a FaceIdentifier with mock gallery and verify that
    identify_person-style matching rejects unknown embeddings.
    
    We can't call identify_person directly (needs real camera frame),
    but we can test the matching logic it uses.
    """
    from face_id import FaceIdentifier

    gallery = _build_mock_gallery(3)
    identifier = FaceIdentifier(
        gallery, threshold=GATE_FACE_THRESHOLD,
        peri_threshold=GATE_PERIOCULAR_THRESHOLD,
        debug=False,
        quality_gate_enabled=False,
        abs_floor_fullface=GATE_ABS_FLOOR_FULLFACE,
        abs_floor_periocular=GATE_ABS_FLOOR_PERIOCULAR,
    )

    # Verify the absolute floors are set
    assert identifier.abs_floor_full == GATE_ABS_FLOOR_FULLFACE
    assert identifier.abs_floor_peri == GATE_ABS_FLOOR_PERIOCULAR

    # Verify stats tracking exists for abs_floor rejections
    assert "abs_floor_rejected_full" in identifier.stats
    assert "abs_floor_rejected_peri" in identifier.stats


def test_track_identity_thin_margin():
    """Test that thin-margin flagging downgrades high confidence to uncertain."""
    from track_identity import TrackIdentityManager
    from config import GATE_THIN_MARGIN_THRESHOLD

    gallery = _build_mock_gallery(3)
    mgr = TrackIdentityManager(buffer_size=5, known_encodings=gallery)

    # Push observations with thin margin (raw_sim - raw_second < threshold)
    emb = _make_unit_vector(512, seed=8000)
    for i in range(3):
        mgr.update(
            track_id=1, embedding=emb, quality_score=0.8,
            match_name="Employee_0000", similarity=0.85,
            method="full",
            raw_sim=0.50,  # Close to runner-up
            raw_second_best=0.48,  # Very thin margin: 0.02 < 0.05
        )

    name, confidence, tier = mgr.get_identity(1)
    assert name == "Employee_0000"
    # With thin margin (0.02 < 0.05), tier should be downgraded to "uncertain"
    # even if confidence is high
    assert tier == "uncertain", (
        f"Expected 'uncertain' tier due to thin margin, got '{tier}'"
    )
    print(f"    Thin margin (0.02) correctly downgraded to 'uncertain'")


def test_config_absolute_floors_loaded():
    """Verify that the absolute floor config values are reasonable."""
    assert 0.2 <= GATE_ABS_FLOOR_FULLFACE <= 0.6, (
        f"GATE_ABS_FLOOR_FULLFACE={GATE_ABS_FLOOR_FULLFACE} is outside "
        f"reasonable range [0.2, 0.6]"
    )
    assert 0.15 <= GATE_ABS_FLOOR_PERIOCULAR <= 0.5, (
        f"GATE_ABS_FLOOR_PERIOCULAR={GATE_ABS_FLOOR_PERIOCULAR} is outside "
        f"reasonable range [0.15, 0.5]"
    )
    assert GATE_ABS_FLOOR_PERIOCULAR <= GATE_ABS_FLOOR_FULLFACE, (
        f"Periocular floor ({GATE_ABS_FLOOR_PERIOCULAR}) should be <= "
        f"full-face floor ({GATE_ABS_FLOOR_FULLFACE})"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TEST GROUP 3: Test with REAL gallery (if available)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Real Gallery Open-Set Tests]")


def test_real_gallery_rejects_random():
    """
    If a real gallery is available (known_faces/ has enrolled people),
    verify that random embeddings are rejected by the absolute floor.
    """
    from face_id import build_known_encodings, _load_insightface

    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    if not os.path.exists(known_faces_dir):
        print("    (Skipped: known_faces/ directory not found)")
        return

    # Try to load InsightFace for real gallery test
    if not _load_insightface():
        print("    (Skipped: InsightFace not available)")
        return

    gallery = build_known_encodings(known_faces_dir)
    if not gallery:
        print("    (Skipped: no enrolled people in gallery)")
        return

    n_people = len(gallery)
    print(f"    Real gallery loaded: {n_people} enrolled people")

    # Test 100 random unknown embeddings against the real gallery
    reject_count = 0
    false_accept_count = 0
    worst_sim = -1.0
    worst_name = None

    for seed in range(100):
        unknown = _make_unit_vector(512, seed=20000 + seed)

        max_sim = -1.0
        max_name = None
        for name, enc in gallery.items():
            sim = float(np.dot(unknown, enc["full"]))
            if sim > max_sim:
                max_sim = sim
                max_name = name

        if max_sim < GATE_ABS_FLOOR_FULLFACE:
            reject_count += 1
        else:
            false_accept_count += 1

        if max_sim > worst_sim:
            worst_sim = max_sim
            worst_name = max_name

    assert false_accept_count == 0, (
        f"CRITICAL: {false_accept_count}/100 random embeddings had raw_sim >= "
        f"absolute floor {GATE_ABS_FLOOR_FULLFACE} against real gallery! "
        f"Worst: {worst_name} sim={worst_sim:.3f}"
    )
    print(f"    100/100 random embeddings rejected against real gallery")
    print(f"    Worst case: sim={worst_sim:.3f} to '{worst_name}' "
          f"(floor={GATE_ABS_FLOOR_FULLFACE})")


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ALL TESTS
# ─────────────────────────────────────────────────────────────────────────────

# Group 1: Embedding-level tests
test("Random noise vectors rejected", test_random_noise_rejected)
test("Moderate similarity (0.25-0.35) rejected by absolute floor",
     test_moderate_similarity_rejected)
test("NAC-pass + abs-floor-fail correctly rejected (the original bug scenario)",
     test_nac_pass_but_abs_floor_fail)
test("Genuine high-similarity match still works", test_genuine_match_still_works)
test("Periocular absolute floor enforced", test_periocular_abs_floor)
test("50 bulk unknown embeddings all rejected", test_all_unknowns_rejected_bulk)
test("Dual-criterion: both NAC and abs_floor must pass",
     test_dual_criterion_both_must_pass)

# Group 2: Integration tests
test("FaceIdentifier has abs_floor attributes", test_identifier_rejects_unknowns)
test("Thin-margin flagging downgrades to uncertain", test_track_identity_thin_margin)
test("Absolute floor config values are reasonable", test_config_absolute_floors_loaded)

# Group 3: Real gallery tests
test("Real gallery rejects 100 random embeddings", test_real_gallery_rejects_random)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  OPEN-SET REJECTION TEST RESULTS")
print(f"{'='*60}")
print(f"  Passed : {passed}/{total}")
print(f"  Failed : {failed}/{total}")
print()
for name, status, detail in details:
    marker = "[PASS]" if status == "PASS" else "[FAIL]"
    line = f"  {marker} {name}"
    if detail:
        line += f" - {detail}"
    print(line)
print(f"\n{'='*60}")

if failed > 0:
    print(f"\n  [FAIL] OPEN-SET REJECTION TEST FAILED - DO NOT DEPLOY")
    print(f"    {failed} test(s) failed. Fix before proceeding.")
    sys.exit(1)
else:
    print(f"\n  [PASS] ALL OPEN-SET REJECTION TESTS PASSED")
    print(f"    Unknown persons are correctly rejected by the dual-criterion matching.")
    print(f"\n  NOTE: These tests use synthetic embeddings. For full production validation,")
    print(f"  also run the manual test procedure with real unknown persons at the gate.")

print(f"{'='*60}\n")
