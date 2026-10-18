"""
tests/test_search_smoke.py
==========================
Offline end-to-end smoke test for the person search pipeline.

Uses cctv.mp4 / cctv2.mp4 from test-footage/ as FAKE camera sources.
No real cameras are opened.

Pipeline tested:
  main.py → CameraManager → IngestWorker → FaceEmbedder →
  ObservationStore → ProbeBuilder → Matcher → search_api

Steps:
  1. Create a CameraManager with file-based sources
  2. Start IngestManager to ingest ~20 seconds of video
  3. Verify observations are stored (RAM + SQLite)
  4. Extract a face from the video as a probe image
  5. Build a probe template bank
  6. Run retro search
  7. Verify results structure
  8. Shut down cleanly — no thread errors

Run:
  .\.python\python.exe tests/test_search_smoke.py
"""

import os
import sys
import time
import shutil
import sqlite3
import traceback
import threading
from pathlib import Path

# Project root
PROJECT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import cv2
import numpy as np

PASS = 0
FAIL = 0
ERRORS = []

def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        ERRORS.append(name)


# Temp directories for this test
TEST_DB = os.path.join(PROJECT, "_smoke_test_obs.db")
TEST_CROPS = os.path.join(PROJECT, "_smoke_test_crops")
TEST_PROBE = os.path.join(PROJECT, "_smoke_test_probe.jpg")
TEST_YAML = os.path.join(PROJECT, "_smoke_test_cameras.yaml")


def cleanup():
    """Remove all test artifacts."""
    for f in [TEST_DB, TEST_PROBE, TEST_YAML]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(TEST_CROPS):
        shutil.rmtree(TEST_CROPS, ignore_errors=True)


def write_test_cameras_yaml():
    """Create a cameras.yaml that uses file sources instead of real cameras."""
    video1 = os.path.join(PROJECT, "test-footage", "cctv.mp4")
    video2 = os.path.join(PROJECT, "test-footage", "cctv2.mp4")

    if not os.path.exists(video1):
        print(f"  SKIP: {video1} not found")
        return False

    # Use video files as "cameras"
    yaml_content = f"""cameras:
  - camera_id: fake_cam_1
    name: "Smoke Test Camera 1"
    type: file
    detect_stream: "{video1.replace(chr(92), '/')}"
    enabled: true
    location_label: "test"

  - camera_id: fake_cam_2
    name: "Smoke Test Camera 2"
    type: file
    detect_stream: "{video2.replace(chr(92), '/')}"
    enabled: true
    location_label: "test"
"""
    with open(TEST_YAML, "w") as f:
        f.write(yaml_content)
    return True


def extract_probe_face(embedder):
    """Extract a face from cctv.mp4 frame to use as a probe photo."""
    video_path = os.path.join(PROJECT, "test-footage", "cctv.mp4")
    cap = cv2.VideoCapture(video_path)

    # Read through frames until we find a face
    for _ in range(300):  # Try up to 300 frames
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = embedder.detect_faces(img_rgb, det_size=(640, 640))

        if faces:
            # Save the frame as a probe image
            cv2.imwrite(TEST_PROBE, frame)
            cap.release()
            return True

    cap.release()
    return False


def main():
    cleanup()
    print(f"\n{'='*60}")
    print(f"  SEARCH PIPELINE SMOKE TEST (offline)")
    print(f"{'='*60}")
    print(f"  Project: {PROJECT}")
    print(f"  Using: test-footage/cctv.mp4, test-footage/cctv2.mp4\n")

    # ─── Step 1: Write test cameras.yaml ─────────────────────────────
    print("Step 1: Camera config")
    ok = write_test_cameras_yaml()
    check("cameras.yaml written", ok)
    if not ok:
        print("  Cannot continue without test videos.")
        return

    # ─── Step 2: Load CameraManager with file sources ────────────────
    print("\nStep 2: CameraManager")
    try:
        from camera_manager import CameraManager, CameraConfig, CameraHealth
        cam_mgr = CameraManager(TEST_YAML, resize_height=720)
        check("CameraManager created", True)
        check("Cameras loaded", len(cam_mgr.cameras) == 2,
              f"got {len(cam_mgr.cameras)}")
    except Exception as e:
        check("CameraManager created", False, str(e))
        traceback.print_exc()
        cleanup()
        return

    # ─── Step 3: Load FaceEmbedder ───────────────────────────────────
    print("\nStep 3: FaceEmbedder")
    try:
        from face_embedder import InsightFaceEmbedder
        embedder = InsightFaceEmbedder(det_size=(320, 320))
        check("InsightFaceEmbedder loaded", True)
        check("embedding_dim is 512", embedder.embedding_dim == 512)
    except Exception as e:
        check("InsightFaceEmbedder loaded", False, str(e))
        traceback.print_exc()
        cleanup()
        return

    # ─── Step 4: Create ObservationStore ─────────────────────────────
    print("\nStep 4: ObservationStore")
    try:
        from observation_store import ObservationStore
        store = ObservationStore(db_path=TEST_DB, crops_dir=TEST_CROPS)
        check("ObservationStore created", True)
        check("Store starts empty", store.observation_count == 0)
        check("SQLite DB exists", os.path.exists(TEST_DB))
    except Exception as e:
        check("ObservationStore created", False, str(e))
        traceback.print_exc()
        cleanup()
        return

    # ─── Step 5: Start cameras ───────────────────────────────────────
    print("\nStep 5: Start cameras")
    try:
        results = cam_mgr.start_all()
        started = sum(1 for v in results.values() if v)
        check(f"Cameras started ({started}/2)", started >= 1)

        # Wait a moment for reader threads to get first frame
        time.sleep(1.0)

        # Check if we can read a frame
        got_frame = False
        for cam_id, cam in cam_mgr.enabled_cameras.items():
            ret, frame, ts = cam.read()
            if ret and frame is not None:
                got_frame = True
                check(f"Frame from {cam_id}",
                      frame.shape[0] > 0,
                      f"shape={frame.shape}")
                break
        check("Got at least one frame", got_frame)
    except Exception as e:
        check("Cameras started", False, str(e))
        traceback.print_exc()

    # ─── Step 6: Start IngestManager ─────────────────────────────────
    print("\nStep 6: IngestManager (ingest ~15 seconds of video)")
    try:
        from ingest_pipeline import IngestManager
        from detector import PersonDetector

        detector_kwargs = dict(
            weights="yolo11n.pt",
            conf=0.3,
            iou=0.5,
            imgsz=480,
            tracker="custom_botsort.yaml",
        )

        ingest = IngestManager(
            cam_mgr, embedder, store,
            detector_kwargs=detector_kwargs,
            detect_fps=5.0, idle_fps=1.0,
        )
        check("IngestManager created", len(ingest.workers) >= 1,
              f"{len(ingest.workers)} workers")

        ingest.start_all()
        check("Ingest started", True)

        # Let it ingest for 15 seconds
        print("  ... ingesting (15 seconds) ...")
        for i in range(15):
            time.sleep(1.0)
            count = store.observation_count
            if count > 0 and i >= 5:
                print(f"  ... {count} observations after {i+1}s, continuing...")
                break

        obs_count = store.observation_count
        check(f"Observations stored in RAM", obs_count > 0,
              f"count={obs_count}")

        # Check SQLite
        conn = sqlite3.connect(TEST_DB)
        db_count = conn.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        conn.close()
        check(f"Observations in SQLite", db_count > 0,
              f"count={db_count}")

        # Print stats
        stats = ingest.stats()
        total_frames = sum(s['frames_processed'] for s in stats.values())
        total_faces = sum(s['faces_detected'] for s in stats.values())
        total_embs = sum(s['embeddings_computed'] for s in stats.values())
        print(f"  Stats: frames={total_frames}, faces={total_faces}, "
              f"embeddings={total_embs}")

    except Exception as e:
        check("IngestManager created", False, str(e))
        traceback.print_exc()

    # ─── Step 7: Extract probe face from video ───────────────────────
    print("\nStep 7: Probe extraction")
    try:
        probe_ok = extract_probe_face(embedder)
        check("Probe face extracted from video", probe_ok)
        if probe_ok:
            check("Probe image saved", os.path.exists(TEST_PROBE))
    except Exception as e:
        check("Probe face extracted", False, str(e))
        traceback.print_exc()

    # ─── Step 8: Build probe template ────────────────────────────────
    print("\nStep 8: ProbeBuilder")
    probe = None
    try:
        from probe_builder import ProbeBuilder
        builder = ProbeBuilder(embedder)

        if os.path.exists(TEST_PROBE):
            probe = builder.build_from_photos([TEST_PROBE])
            check("Probe built", probe is not None)
            if probe:
                check(f"Template bank shape",
                      probe.template_bank.shape[1] == 512,
                      f"shape={probe.template_bank.shape}")
                check(f"Templates generated",
                      probe.template_count >= 1,
                      f"count={probe.template_count}")
                check(f"Quality report",
                      probe.quality_report.reliability in ("good", "moderate", "weak"),
                      f"reliability={probe.quality_report.reliability}")
                if probe.quality_report.warnings:
                    for w in probe.quality_report.warnings:
                        print(f"    ⚠ {w}")
        else:
            check("Probe built", False, "No probe image")
    except Exception as e:
        check("Probe built", False, str(e))
        traceback.print_exc()

    # ─── Step 9: Run retro search ────────────────────────────────────
    print("\nStep 9: Matcher retro search")
    try:
        from matcher import Matcher, MatchTier

        if probe is not None and obs_count > 0:
            # Get matrix snapshot
            matrix, ids, meta = store.get_matrix_snapshot()
            check("Matrix snapshot", matrix is not None,
                  f"shape={matrix.shape if matrix is not None else None}")

            # Run matcher
            matcher = Matcher(store)
            result = matcher.retro_search(probe, last_minutes=60)

            check("Search result returned", result is not None)
            check("Search has coverage", 'cameras_searched' in result.coverage)
            check("Search timestamp set", len(result.timestamp) > 0)

            n_matches = len(result.all_matches)
            n_episodes = len(result.episodes)
            print(f"  Results: {n_matches} matches in {n_episodes} episodes")

            if n_matches > 0:
                best = result.all_matches[0]
                print(f"  Best match: score={best.combined_score:.3f} "
                      f"tier={best.tier} camera={best.camera_id}")
                check("Match has valid tier",
                      best.tier in (MatchTier.STRONG, MatchTier.POSSIBLE,
                                    MatchTier.WEAK))
                check("Match has valid score",
                      0 <= best.combined_score <= 1.0)

            # Check audit log
            conn = sqlite3.connect(TEST_DB)
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM search_audit"
            ).fetchone()[0]
            conn.close()
            check("Search logged to audit", audit_count > 0)
        else:
            print("  SKIP: No probe or no observations")
    except Exception as e:
        check("Retro search", False, str(e))
        traceback.print_exc()

    # ─── Step 10: Live match single embedding ────────────────────────
    print("\nStep 10: Live match (single embedding)")
    try:
        if probe is not None and matrix is not None:
            test_emb = matrix[0]  # Use first observation embedding
            live_result = matcher.live_match(
                probe, test_emb, None, "fake_cam_1", time.time()
            )
            # live_result can be None (below noise floor) or a MatchResult
            check("Live match returned", True,
                  f"result={'hit' if live_result else 'below noise'}")
    except Exception as e:
        check("Live match", False, str(e))
        traceback.print_exc()

    # ─── Step 11: CLI search formatting ──────────────────────────────
    print("\nStep 11: CLI search output (dry run)")
    try:
        from search_api import _print_results
        if probe is not None:
            _print_results(result)
            check("CLI output formatted", True)
    except Exception as e:
        check("CLI output", False, str(e))
        traceback.print_exc()

    # ─── Step 12: Clean shutdown ─────────────────────────────────────
    print("\nStep 12: Clean shutdown")
    try:
        ingest.stop_all()
        check("Ingest stopped", True)

        cam_mgr.stop_all()
        check("Cameras stopped", True)

        # Check for leftover non-daemon threads
        time.sleep(0.5)
        alive_threads = [t for t in threading.enumerate()
                         if t.is_alive() and not t.daemon
                         and t.name != "MainThread"]
        check("No leftover non-daemon threads",
              len(alive_threads) == 0,
              f"alive: {[t.name for t in alive_threads]}")

        # Print final stats
        ingest.print_stats()

    except Exception as e:
        check("Clean shutdown", False, str(e))
        traceback.print_exc()

    # ─── Summary ─────────────────────────────────────────────────────
    cleanup()

    print(f"\n{'='*60}")
    if FAIL == 0:
        print(f"  ALL {PASS} CHECKS PASSED")
    else:
        print(f"  {PASS} passed, {FAIL} FAILED")
        for e in ERRORS:
            print(f"    ✗ {e}")
    print(f"{'='*60}\n")

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
