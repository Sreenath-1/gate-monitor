"""
test_pipeline.py
================
Offline static tests for the gate monitor pipeline.

Tests everything that can be verified WITHOUT a live person in front of the camera:
  1. Module imports (all dependencies available)
  2. Config loading (.env, argparse)
  3. Video source opening (cctv.mp4)
  4. YOLO model loading and inference
  5. Tracker operation
  6. Zone crossing logic (synthetic trajectories)
  7. SQLite logger (schema, insert, query)
  8. Timestamp timezone verification (IST, not UTC)
  9. Debounce logic

Does NOT test: live camera detection, real person crossing, face recognition accuracy.
Those require manual testing with a person physically in frame.

Run:
    python scripts/test_pipeline_offline.py
"""

import sys, os, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
total = 0


def test(name, func):
    global passed, failed, total
    total += 1
    try:
        func()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 1: Module Imports
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  OFFLINE PIPELINE TESTS")
print(f"{'='*60}\n")

print("  [Module Imports]")

def test_import_config():
    from config import parse_args, IST, CONF_THRESHOLD, DEBOUNCE_FRAMES
    assert IST is not None, "IST timezone not defined"
    assert DEBOUNCE_FRAMES >= 1, f"DEBOUNCE_FRAMES should be >= 1, got {DEBOUNCE_FRAMES}"

def test_import_stream():
    from stream import FrameSource, ThreadedCapture, check_connection, is_stream_source
    assert is_stream_source("rtsp://test") == True
    assert is_stream_source("test-footage/cctv.mp4") == False
    assert is_stream_source("0") == False

def test_import_detector():
    from detector import PersonDetector, Detection, FrameResult

def test_import_zone():
    from zone import CrossingZone, CrossingEvent

def test_import_logger():
    from logger import EventLogger

def test_import_visualizer():
    from visualizer import fmt_duration, fmt_time, draw_gate_line

test("Import config", test_import_config)
test("Import stream", test_import_stream)
test("Import detector", test_import_detector)
test("Import zone", test_import_zone)
test("Import logger", test_import_logger)
test("Import visualizer", test_import_visualizer)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 2: .env Loading
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [.env Configuration]")

def test_env_loading():
    rtsp_url = os.environ.get("RTSP_URL")
    assert rtsp_url is not None, "RTSP_URL not loaded from .env"
    assert "rtsp://" in rtsp_url.lower(), f"RTSP_URL doesn't look like an RTSP URL: {rtsp_url}"
    print(f"    RTSP_URL = {rtsp_url}")

test(".env loads RTSP_URL", test_env_loading)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 3: Video Source
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Video Source]")

def test_video_opens():
    from stream import FrameSource
    src = FrameSource("test-footage/cctv.mp4", resize_height=0)
    info = src.info_dict()
    assert src.raw_width > 0, "Width is 0"
    assert src.raw_height > 0, "Height is 0"
    ret, frame = src.read()
    assert ret, "First frame read failed"
    assert frame is not None, "Frame is None"
    src.release()
    print(f"    {info['raw_resolution']}, {info['fps']} FPS, {info['total_frames']} frames")

test("cctv.mp4 opens and reads", test_video_opens)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 4: YOLO Model Loading + Inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [YOLO Detection]")

def test_yolo_loads():
    from detector import PersonDetector
    det = PersonDetector("yolo11n.pt", imgsz=480)
    assert det.model is not None, "Model is None"

def test_yolo_inference():
    import cv2
    from detector import PersonDetector
    det = PersonDetector("yolo11n.pt", imgsz=480)
    cap = cv2.VideoCapture("test-footage/cctv.mp4")
    # Skip to a frame with a person (frame 30 based on direction analysis)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read frame 30"

    result = det.detect_no_track(frame)
    print(f"    Frame 30: {len(result.detections)} detections, "
          f"{result.inference_ms:.0f}ms inference")
    assert len(result.detections) > 0, "No person detected in frame 30 (expected at least 1)"

test("YOLO model loads", test_yolo_loads)
test("YOLO detects person in frame 30", test_yolo_inference)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 5: Tracking
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Tracking]")

def test_tracking():
    import cv2
    from detector import PersonDetector
    det = PersonDetector("yolo11n.pt", tracker="botsort.yaml", imgsz=480)
    cap = cv2.VideoCapture("test-footage/cctv.mp4")
    track_ids_seen = set()
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        result = det.detect(frame)
        for d in result.detections:
            track_ids_seen.add(d.track_id)
    cap.release()
    print(f"    50 frames: {len(track_ids_seen)} unique track IDs: {sorted(track_ids_seen)}")

test("BoT-SORT tracking on 50 frames", test_tracking)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 6: Zone Crossing Logic (Synthetic)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Zone Crossing]")

def test_zone_basic_crossing():
    from zone import CrossingZone
    zone = CrossingZone(
        direction="horizontal", line_position=0.5,
        frame_width=640, frame_height=480,
        flip=False, min_track_duration=0.0, debounce_frames=1
    )
    # Simulate person moving downward (exit)
    zone.update(1, 320, 200)  # above line (240)
    event = zone.update(1, 320, 260)  # below line
    assert event is not None, "Expected exit event"
    assert event.event_type == "exit", f"Expected 'exit', got '{event.event_type}'"

    # Simulate person moving back up (entry)
    event = zone.update(1, 320, 220)  # above line again
    assert event is not None, "Expected entry event"
    assert event.event_type == "entry", f"Expected 'entry', got '{event.event_type}'"

def test_zone_debounce():
    from zone import CrossingZone
    zone = CrossingZone(
        direction="horizontal", line_position=0.5,
        frame_width=640, frame_height=480,
        flip=False, min_track_duration=0.0, debounce_frames=3
    )
    # Frame 1: above line
    zone.update(1, 320, 200)

    # Frame 2: below line — pending exit starts
    event = zone.update(1, 320, 260)
    assert event is None, "Should not fire immediately with debounce=3"

    # Frame 3: still below — pending at count 2
    event = zone.update(1, 320, 265)
    assert event is None, "Should not fire at count 2 with debounce=3"

    # Frame 4: still below — debounce met at count 3
    event = zone.update(1, 320, 270)
    assert event is not None, "Should fire at count 3"
    assert event.event_type == "exit"

def test_zone_debounce_cancel():
    from zone import CrossingZone
    zone = CrossingZone(
        direction="horizontal", line_position=0.5,
        frame_width=640, frame_height=480,
        flip=False, min_track_duration=0.0, debounce_frames=3
    )
    # Person above line
    zone.update(1, 320, 200)

    # Crosses below
    zone.update(1, 320, 260)

    # Bounces back above — should cancel
    event = zone.update(1, 320, 200)
    assert event is None, "Should cancel pending crossing when bouncing back"

    # Cross below again, complete debounce
    zone.update(1, 320, 260)
    zone.update(1, 320, 265)
    event = zone.update(1, 320, 270)
    assert event is not None, "Should fire after re-crossing with full debounce"

def test_zone_min_track_duration():
    from zone import CrossingZone
    zone = CrossingZone(
        direction="horizontal", line_position=0.5,
        frame_width=640, frame_height=480,
        flip=False, min_track_duration=1.5, debounce_frames=1
    )
    # Brand new track crosses — should be ignored (track too young)
    zone.update(1, 320, 200)
    event = zone.update(1, 320, 260)
    assert event is None, "Should not fire for track younger than min_track_duration"

test("Basic crossing detection", test_zone_basic_crossing)
test("Debounce (3 frames required)", test_zone_debounce)
test("Debounce cancellation on bounce-back", test_zone_debounce_cancel)
test("Min track duration filter", test_zone_min_track_duration)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 7: SQLite Logger
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [SQLite Logger]")

def test_logger_schema():
    from logger import EventLogger
    db_path = os.path.join(os.path.dirname(__file__), "test_gate_events.db")
    try:
        logger = EventLogger(db_path)
        # Verify tables exist
        tables = logger.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "crossing_events" in table_names, "crossing_events table missing"
        assert "sessions" in table_names, "sessions table missing"
        logger.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_logger_exit_entry_cycle():
    from logger import EventLogger
    from datetime import datetime
    from config import IST

    db_path = os.path.join(os.path.dirname(__file__), "test_gate_events.db")
    try:
        logger = EventLogger(db_path)
        t1 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=IST)
        t2 = datetime(2025, 1, 15, 14, 35, 0, tzinfo=IST)

        sid = logger.log_exit(5, person_name="TestPerson", timestamp=t1)
        assert sid > 0, f"session_id should be > 0, got {sid}"

        duration = logger.log_entry(5, person_name="TestPerson", timestamp=t2)
        assert duration is not None, "Duration should not be None"
        assert abs(duration - 300.0) < 1, f"Expected ~300s duration, got {duration}"

        sessions = logger.get_sessions()
        assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"
        assert sessions[0]["status"] == "returned"
        assert sessions[0]["person_name"] == "TestPerson"

        # Verify timezone in stored timestamp
        stored_exit = sessions[0]["exit_time"]
        assert "+05:30" in stored_exit, f"Expected IST offset in timestamp, got: {stored_exit}"

        logger.close()
        print(f"    Exit→Entry cycle: {duration:.0f}s duration, status=returned, tz=IST ✓")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_logger_csv_export():
    from logger import EventLogger
    from datetime import datetime
    from config import IST

    db_path = os.path.join(os.path.dirname(__file__), "test_gate_events.db")
    csv_path = os.path.join(os.path.dirname(__file__), "test_export.csv")
    try:
        logger = EventLogger(db_path)
        t1 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=IST)
        t2 = datetime(2025, 1, 15, 14, 35, 0, tzinfo=IST)
        logger.log_exit(5, person_name="CSVTest", timestamp=t1)
        logger.log_entry(5, person_name="CSVTest", timestamp=t2)
        logger.export_csv(csv_path)
        assert os.path.exists(csv_path), "CSV file not created"

        with open(csv_path, "r") as f:
            content = f.read()
        assert "CSVTest" in content, "Person name not in CSV"
        assert "IST" in content, "Timezone not mentioned in CSV"
        logger.close()
    finally:
        for p in [db_path, csv_path]:
            if os.path.exists(p):
                os.remove(p)

test("Schema creates correctly", test_logger_schema)
test("Exit→Entry cycle with IST timestamps", test_logger_exit_entry_cycle)
test("CSV export", test_logger_csv_export)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 8: Timezone Verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Timezone]")

def test_timezone_ist():
    from datetime import datetime
    from config import IST
    now = datetime.now(IST)
    iso = now.isoformat()
    assert "+05:30" in iso, f"Expected IST offset (+05:30) in {iso}"
    print(f"    Current IST: {iso}")

test("Timestamps use IST (UTC+05:30)", test_timezone_ist)


# ─────────────────────────────────────────────────────────────────────────────
#  TEST 9: Track Identity Swap Splitting
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Track ID Splitting]")

def test_identity_swap_splitting():
    from main import get_logical_track
    from logger import EventLogger
    from datetime import datetime
    from config import IST
    
    db_path = os.path.join(os.path.dirname(__file__), "test_swap.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        
    try:
        logger = EventLogger(db_path)
        track_mapping = {}
        track_names = {}
        next_logical_tid_ref = [1_000_000]
        now = datetime.now(IST)
        
        # 1. Setup: Alice is recognized as Track 5 and exits (logs an open session)
        tid1 = get_logical_track(5, "Alice", track_mapping, track_names, next_logical_tid_ref, logger, now)
        track_names[tid1] = "Alice"
        assert tid1 == 5
        sid = logger.log_exit(tid1, person_name="Alice", timestamp=now)
        
        # 2. He is tracked again in a later frame (still Alice)
        tid2 = get_logical_track(5, "Alice", track_mapping, track_names, next_logical_tid_ref, logger, now)
        assert tid2 == 5  # No swap, should remain 5
        
        # 3. SWAP! The track freezes, and Bob gets attached to Track 5
        tid3 = get_logical_track(5, "Bob", track_mapping, track_names, next_logical_tid_ref, logger, now)
        track_names[tid3] = "Bob"
        
        assert tid3 == 1_000_000, f"Expected split ID 1000000, got {tid3}"
        assert track_mapping[5] == 1_000_000, "Mapping not updated"
        
        # 4. Verify original session was closed
        sessions = logger.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["status"] == "lost", f"Original session should be 'lost', got {sessions[0]['status']}"
        assert sessions[0]["person_name"] == "Alice"
        
        logger.close()
        print("    Identity swap correctly split the track and closed orphaned session ✓")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

test("Identity swap splits logical track", test_identity_swap_splitting)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*60}\n")

if failed > 0:
    sys.exit(1)
