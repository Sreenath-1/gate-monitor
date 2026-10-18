"""
main.py
=======
Pipeline orchestrator for the Gate Monitor system.

Operational modes:
  --mode attendance    Original warehouse entry/exit detection (default)
  --mode search        "Find this person" open-set face search (Stages 2-5)
  --mode sources-check Camera diagnostics — test all cameras for N seconds

Attendance mode composes all modules:
  config.py    → CLI args + constants + .env loading
  stream.py    → Video/RTSP source handling (with auto-reconnect)
  detector.py  → YOLOv11 person detection + tracking (with imgsz optimization)
  zone.py      → Entry/exit crossing detection (with debounce)
  logger.py    → SQLite event logging (IST timestamps)
  visualizer.py → OpenCV drawing functions
  face_id.py   → (optional) Face recognition

Search mode (Stages 2+) uses:
  camera_manager.py → Multi-camera source management
  (future stages)   → Embedding store, probe builder, matcher, search UI

Usage:
    # Attendance (default — unchanged behavior):
    python main.py --source cctv.mp4
    python main.py --source rtsp

    # Camera diagnostics:
    python main.py --mode sources-check

    # Person search (Stages 2+):
    python main.py --mode search
"""

import sys
import os
import torch  # MUST be imported early on Windows to set CUDA DLL paths
import time
import signal

import cv2
import numpy as np
from datetime import datetime

# Ensure this directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    parse_args, IST, C_ENTER, C_EXIT, C_ALERT,
    VIDEO_CHUNK_MINUTES, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD,
    GATE_THIN_MARGIN_THRESHOLD, DEFAULT_DAILY_ATTENDANCE_CSV,
)
from stream import FrameSource
from detector import PersonDetector
from zone import CrossingZone
from logger import EventLogger
from visualizer import (
    fmt_duration, draw_gate_line, draw_person_box, draw_centroid,
    draw_left_panel, draw_top_bar, draw_event_flash,
)

# Stage 7: Track-level embedding fusion
try:
    from track_identity import TrackIdentityManager
    _TRACK_ID_AVAILABLE = True
except ImportError:
    _TRACK_ID_AVAILABLE = False

# Stage 7: Structured audit logging
try:
    from audit_log import get_audit_logger
    _AUDIT_LOG_AVAILABLE = True
except ImportError:
    _AUDIT_LOG_AVAILABLE = False

# Optional face recognition — gracefully disabled if not installed
try:
    from face_id import build_known_encodings, FaceIdentifier
    _FACE_ID_AVAILABLE = True
except Exception:
    _FACE_ID_AVAILABLE = False


def _now():
    """Get current time in IST."""
    return datetime.now(IST)


def get_logical_track(raw_tid, matched_name, track_mapping, track_names, next_logical_tid_ref, event_logger, now):
    """Map raw ultralytics track ID to a logical track ID, handling identity swaps."""
    if raw_tid not in track_mapping:
        track_mapping[raw_tid] = raw_tid
    logical_tid = track_mapping[raw_tid]
    
    if matched_name:
        old_name = track_names.get(logical_tid)
        # Check for an identity swap (e.g., Alice -> Sivaraju)
        if old_name and not old_name.startswith("Person #") and old_name != matched_name:
            print(f"  [WARNING] Track ID {raw_tid} swapped identity from {old_name} to {matched_name}! Splitting session...")
            if event_logger:
                event_logger.close_session(logical_tid, status="lost", timestamp=now)
            # Split the track: assign a new logical ID
            logical_tid = next_logical_tid_ref[0]
            next_logical_tid_ref[0] += 1
            track_mapping[raw_tid] = logical_tid
    return logical_tid


def main_attendance():
    """Original warehouse entry/exit detection mode (--mode attendance)."""
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WAREHOUSE ENTRY/EXIT DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"  Source       : {args.source}")
    print(f"  Weights      : {args.weights}")
    print(f"  Tracker      : {args.tracker}")
    print(f"  YOLO imgsz   : {args.imgsz}  (inference resolution)")
    print(f"  Gate line    : {args.line*100:.0f}% ({args.direction})")
    print(f"  Min track    : {args.min_track_duration}s (anti-flicker)")
    print(f"  Debounce     : {args.debounce_frames} frames (crossing confirmation)")
    print(f"  Overstay     : {args.overstay} min")
    print(f"  Direction    : {'flipped' if args.flip else 'normal'}")
    print(f"  Timezone     : IST (UTC+05:30)")
    print(f"  Database     : {args.db}")
    print(f"{'='*60}\n")

    # ── Stage 7: Initialize audit logger ─────────────────────────────────
    audit_logger = None
    if _AUDIT_LOG_AVAILABLE:
        audit_logger = get_audit_logger(args.debug_log)
        print(f"  Structured debug log: {args.debug_log}")

    # ── Stage 7: Initialize track identity manager ─────────────────────
    track_identity_mgr = None

    # ── Load face recognition (optional) ──────────────────────────────────
    face_id = None
    if _FACE_ID_AVAILABLE:
        print("  Loading face recognition (InsightFace ArcFace)...")
        known_encodings = build_known_encodings(args.faces_dir)
        if known_encodings:
            # Stage 7: Create track identity manager with known encodings
            if _TRACK_ID_AVAILABLE:
                track_identity_mgr = TrackIdentityManager(
                    known_encodings=known_encodings
                )

            face_id = FaceIdentifier(
                known_encodings,
                match_every_n=args.face_every,
                threshold=args.face_threshold,
                peri_threshold=args.peri_threshold,
                quality_gate_enabled=args.quality_gate,
                track_identity_mgr=track_identity_mgr,
            )
            n = len(known_encodings)
            qg_status = "ON" if args.quality_gate else "OFF"
            print(f"  Face recognition: ENABLED ({n} known {'person' if n==1 else 'people'}, "
                  f"every {args.face_every} frames, quality_gate={qg_status})\n")
        else:
            print("  Face recognition: DISABLED (no valid photos found)\n")
    else:
        print("  Face recognition: DISABLED (insightface not installed)\n")

    # ── Load detector ─────────────────────────────────────────────────────
    print(f"  Loading {args.weights} with {args.tracker} tracker (imgsz={args.imgsz})...")
    detector = PersonDetector(
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    print(f"  Detector ready.\n")

    # ── Warmup models to prevent first-frame latency spikes ───────────────
    print("  Warming up models to prevent initial latency spikes...")
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    if face_id:
        import glob
        known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
        face_images = glob.glob(os.path.join(known_faces_dir, "**", "*.[jp][pn]*"), recursive=True)
        if face_images:
            img = cv2.imread(face_images[0])
            if img is not None:
                dummy_img = cv2.resize(img, (1280, 720))
    detector.model.track(dummy_img, persist=True, tracker=args.tracker, verbose=False)
    if face_id:
        face_id.identify_person(dummy_img, 0, 0, dummy_img.shape[1], dummy_img.shape[0], -1, -1)
    print("  Warmup complete.\n")

    # ── Open source ───────────────────────────────────────────────────────
    print(f"  Opening source: {args.source}")
    source = FrameSource(args.source, resize_height=args.resize)
    info = source.info_dict()
    print(f"  Resolution   : {info['raw_resolution']} → {info['process_resolution']}")
    print(f"  FPS          : {info['fps']}")
    print(f"  Total frames : {info['total_frames']}\n")

    # ── Initialize zone ───────────────────────────────────────────────────
    zone = CrossingZone(
        direction=args.direction,
        line_position=args.line,
        frame_width=source.frame_width,
        frame_height=source.frame_height,
        flip=args.flip,
        min_track_duration=args.min_track_duration,
        debounce_frames=args.debounce_frames,
    )
    print(f"  Gate line at pixel: {zone.line_coord} "
          f"({'Y' if args.direction == 'horizontal' else 'X'} axis)")
    print(f"  Crossing debounce: {args.debounce_frames} consecutive frames required")

    # ── Initialize logger ─────────────────────────────────────────────────
    event_logger = None
    if not args.no_output:
        event_logger = EventLogger(args.db)
        print(f"  Database     : {args.db} (timestamps in IST)")

    # ── Video writer ──────────────────────────────────────────────────────
    enable_recording = (not source.is_stream or args.record) and not args.no_output
    writer = None
    chunk_start_time = time.time()
    chunk_index = 0

    def create_writer(path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, source.raw_fps,
                               (source.frame_width, source.frame_height))

    if enable_recording:
        if source.is_stream:
            base, ext = os.path.splitext(args.output)
            chunk_path = f"{base}_chunk{chunk_index:03d}{ext}"
            writer = create_writer(chunk_path)
            print(f"  📹 Recording: {chunk_path}")
        else:
            writer = create_writer(args.output)
            print(f"  📹 Recording: {args.output}")

    # ── State ─────────────────────────────────────────────────────────────
    track_names = {}          # track_id → display name
    track_mapping = {}        # ultralytics_tid → logical_tid
    next_logical_tid_ref = [1_000_000] # Pass by reference list
    flash_events = []         # [(text, color, expiry_frame)]
    total_exits = 0
    total_entries = 0
    alert_count = 0
    alerted_ids = set()       # Track IDs already alerted for overstay
    frame_idx = 0
    t_start = time.time()

    # ── Graceful shutdown ─────────────────────────────────────────────────
    shutdown_requested = False
    def handle_signal(signum, frame_):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\n  ⚠ Shutdown requested (Ctrl+C). Saving data...")
    signal.signal(signal.SIGINT, handle_signal)

    print(f"\n  Monitoring gate... (Press ESC to stop)\n")

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═════════════════════════════════════════════════════════════════════
    try:
        while not shutdown_requested:
            # Check for permanent connection loss (RTSP)
            if source.connection_lost:
                print("\n  ✗ Camera connection permanently lost. Shutting down.")
                break

            active_logical_tids = set()

            ret, frame = source.read()
            if not ret:
                if not source.is_stream:
                    break  # Video file ended
                time.sleep(0.05)
                continue

            frame_idx += 1
            now = _now()

            t0 = time.time()
            # ── Detect + Track ────────────────────────────────────────────────
            result = detector.detect(frame)
            t1 = time.time()

            # ── Process each tracked person ───────────────────────────────────
            for det in result.detections:
                raw_tid = det.track_id

                # We need the logical_tid to pass to face_id cache so it tracks failures across fragmentation.
                # get_logical_track handles identity swaps based on matched_name. 
                # If matched_name is None, it just maps raw_tid to its existing logical track.
                logical_tid = get_logical_track(raw_tid, None, track_mapping, track_names, next_logical_tid_ref, event_logger, now)
                
                # Face recognition (optional, runs every face_every frames)
                matched_name = None
                if face_id:
                    # Priority check: if centroid is very close to the gate line, or if they are in the debounce zone, 
                    # force an immediate face ID check to ensure they are evaluated exactly at crossing.
                    coord = zone._get_coord(det.cx, det.cy)
                    distance_to_line = abs(coord - zone.line_coord)
                    is_priority = (distance_to_line < 100) or (logical_tid in zone._pending)
                    
                    matched_name = face_id.identify_person(
                        frame, det.x1, det.y1, det.x2, det.y2, logical_tid, frame_idx, priority=is_priority
                    )
                    
                # If we matched a new name, get_logical_track will handle the identity swap
                if matched_name:
                    tid = get_logical_track(raw_tid, matched_name, track_mapping, track_names, next_logical_tid_ref, event_logger, now)
                else:
                    tid = get_logical_track(raw_tid, None, track_mapping, track_names, next_logical_tid_ref, event_logger, now)

                active_logical_tids.add(tid)
                source.last_seen_frame = getattr(source, 'last_seen_frame', {})
                source.last_seen_frame[tid] = frame_idx

                display_name = matched_name or track_names.get(tid, f"Person #{tid}")
                track_names[tid] = display_name

                # Stage 7: Update bbox for tailgating detection
                zone.update_bbox(tid, det.x1, det.y1, det.x2, det.y2)

                # ── Check crossing (with debounce) ────────────────────────────
                event = zone.update(tid, det.cx, det.cy)

                if event:
                    # Stage 7: Confidence-tiered decision arbitration
                    confidence = 0.0
                    decision_tier = "unresolved"
                    needs_review = False

                    if track_identity_mgr and track_identity_mgr.has_observations(tid):
                        id_name, confidence, decision_tier = track_identity_mgr.get_identity(tid)
                        # Use fused identity if available and better than per-frame
                        if id_name and decision_tier in ("high", "uncertain"):
                            display_name = id_name
                            track_names[tid] = display_name
                        if decision_tier == "uncertain":
                            needs_review = True
                    elif display_name.startswith("Person #"):
                        # No quality-gate-passing recognized frames ever
                        decision_tier = "unresolved"
                        confidence = 0.0
                    else:
                        # We have a name from per-frame matching but no track mgr
                        decision_tier = "high"
                        confidence = 0.5

                    # Stage 7: Tailgating detection
                    tailgaters = zone.check_tailgate(tid, result.active_track_ids)
                    possible_tailgate = len(tailgaters) > 0
                    event.possible_tailgate = possible_tailgate
                    event.tailgate_partner_ids = tailgaters

                    if possible_tailgate and audit_logger:
                        audit_logger.log_tailgate_detected(
                            tid, tailgaters, 0.0, 0
                        )

                    if event.event_type == "exit":
                        total_exits += 1
                        if event_logger:
                            event_logger.log_exit(
                                tid, person_name=display_name,
                                timestamp=now,
                                needs_review=needs_review,
                                possible_tailgate=possible_tailgate,
                            )
                        flash_events.append((
                            f"EXIT — {display_name}  [{now.strftime('%H:%M:%S')}]",
                            C_EXIT,
                            frame_idx + int(source.raw_fps * 2.5),
                        ))
                        tier_tag = f" [{decision_tier}]" if decision_tier != "high" else ""
                        tailgate_tag = " ⚠TAILGATE" if possible_tailgate else ""
                        print(f"  ➜ EXIT   [{now.strftime('%H:%M:%S')}]  {display_name}{tier_tag}{tailgate_tag}")

                    elif event.event_type == "entry":
                        total_entries += 1
                        duration = None
                        if event_logger:
                            duration = event_logger.log_entry(
                                tid, person_name=display_name,
                                timestamp=now,
                                needs_review=needs_review,
                                possible_tailgate=possible_tailgate,
                            )
                        dur_str = fmt_duration(duration) if duration else "?"
                        flash_events.append((
                            f"RETURNED — {display_name}  [{dur_str} outside]",
                            C_ENTER,
                            frame_idx + int(source.raw_fps * 2.5),
                        ))
                        tier_tag = f" [{decision_tier}]" if decision_tier != "high" else ""
                        tailgate_tag = " ⚠TAILGATE" if possible_tailgate else ""
                        print(f"  ← RETURN [{now.strftime('%H:%M:%S')}]  "
                              f"{display_name}  (outside {dur_str}){tier_tag}{tailgate_tag}")

                    # Stage 7: Decision audit log
                    if event_logger:
                        crossing_event_id = event_logger.get_last_crossing_event_id(
                            tid, event.event_type
                        )
                        quality_stats = (
                            track_identity_mgr.get_quality_stats(tid)
                            if track_identity_mgr else
                            {"frames_considered": 0, "frames_passed_quality": 0,
                             "rejection_reasons": {}}
                        )
                        event_logger.log_decision_audit(
                            crossing_event_id=crossing_event_id,
                            track_id=tid,
                            frames_considered=quality_stats["frames_considered"],
                            frames_passed_quality=quality_stats["frames_passed_quality"],
                            rejection_reason_histogram=quality_stats["rejection_reasons"],
                            final_confidence=confidence,
                            decision_tier=decision_tier,
                            possible_tailgate=possible_tailgate,
                        )

                    # Stage 7: Structured crossing log
                    if audit_logger:
                        audit_logger.log_crossing_event(
                            tid, event.event_type, display_name,
                            confidence, decision_tier, possible_tailgate
                        )

                # Update person name in logger if face was recognized after exit
                if matched_name and event_logger:
                    event_logger.update_person_name(tid, matched_name)

                # ── Determine display state ───────────────────────────────────
                is_outside = zone.is_outside(tid)
                secs_out = 0
                if is_outside and event_logger:
                    exit_time = event_logger.get_exit_time(tid)
                    if exit_time:
                        secs_out = (now - exit_time).total_seconds()

                is_overstay = is_outside and secs_out >= args.overstay * 60

                # Overstay alert
                if is_overstay and tid not in alerted_ids:
                    alerted_ids.add(tid)
                    alert_count += 1
                    print(f"  ⚠  OVERSTAY ALERT!  {display_name}  "
                          f"outside for {fmt_duration(secs_out)}")
                    flash_events.append((
                        f"⚠ OVERSTAY — {display_name}  ({fmt_duration(secs_out)})",
                        C_ALERT,
                        frame_idx + int(source.raw_fps * 4),
                    ))

                # Box color
                if is_overstay:
                    color = C_ALERT
                elif is_outside:
                    color = C_EXIT
                else:
                    color = C_ENTER

                # Labels
                if is_outside:
                    label    = f"{display_name}  [OUTSIDE]"
                    sublabel = fmt_duration(secs_out) + " out"
                else:
                    label    = f"{display_name}  [INSIDE]"
                    sublabel = ""

                draw_person_box(frame, det.x1, det.y1, det.x2, det.y2,
                                label, sublabel, color)
                draw_centroid(frame, det.cx, det.cy, color)

            # Stage 7: Update tailgate counters for all active pairs near gate
            zone.update_tailgate_frame(result.active_track_ids)

            # ── Prune stale tracking state ────────────────────────────────────
            # Only prune tracks that haven't been seen for 30 frames (1+ seconds)
            # This prevents 1-frame tracker drops from destroying face caches and crossing states
            stale_tids = set()
            if hasattr(source, 'last_seen_frame'):
                stale_tids = {tid for tid, last_f in source.last_seen_frame.items() 
                              if frame_idx - last_f > 30}
            
            alive_tids = {tid for tid in track_names if tid not in stale_tids}

            if face_id:
                face_id.clear_stale(alive_tids)

            zone.prune_stale(alive_tids)

            for tid in stale_tids:
                if tid in track_names:
                    del track_names[tid]
                if hasattr(source, 'last_seen_frame'):
                    del source.last_seen_frame[tid]

            # ── Overstay check for persons not in frame ───────────────────────
            for tid in zone.outside_set:
                if tid not in result.active_track_ids and tid not in alerted_ids:
                    if event_logger:
                        exit_time = event_logger.get_exit_time(tid)
                        if exit_time:
                            secs_out = (now - exit_time).total_seconds()
                            if secs_out >= args.overstay * 60:
                                alerted_ids.add(tid)
                                alert_count += 1
                                print(f"  ⚠  OVERSTAY ALERT  Person #{tid}  "
                                      f"{fmt_duration(secs_out)} outside (not in frame)")

            # ── Draw overlays ─────────────────────────────────────────────────
            draw_gate_line(frame, zone)

            # Left panel
            if event_logger:
                sessions = event_logger.get_all_sessions_for_display()
                draw_left_panel(frame, sessions, zone.outside_set, args.overstay)

            # Top bar
            elapsed = time.time() - t_start
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            draw_top_bar(frame, total_exits, zone.outside_count, alert_count, live_fps)

            # Flash events
            active_flashes = []
            for text, color, expiry in flash_events:
                if frame_idx <= expiry:
                    draw_event_flash(frame, text, color)
                    active_flashes.append((text, color, expiry))
            flash_events[:] = active_flashes

            # ── Progress reporting ────────────────────────────────────────────
            if source.is_stream:
                if frame_idx % 250 == 0:
                    print(f"  ● Live | {frame_idx} frames ({elapsed:.0f}s)  |  "
                          f"Exits:{total_exits}  Outside:{zone.outside_count}  "
                          f"Alerts:{alert_count}  FPS:{live_fps:.1f}")
            elif frame_idx % 100 == 0 and source.total_frames > 0:
                pct = frame_idx / source.total_frames * 100
                print(f"  Frame {frame_idx}/{source.total_frames} ({pct:.0f}%)  |  "
                      f"Exits:{total_exits}  Outside:{zone.outside_count}  "
                      f"Alerts:{alert_count}  FPS:{live_fps:.1f}")

            # ── Record ────────────────────────────────────────────────────────
            if writer is not None:
                writer.write(frame)
                # Chunked rotation for streams
                if source.is_stream and enable_recording:
                    if (time.time() - chunk_start_time) >= VIDEO_CHUNK_MINUTES * 60:
                        writer.release()
                        chunk_index += 1
                        base, ext = os.path.splitext(args.output)
                        chunk_path = f"{base}_chunk{chunk_index:03d}{ext}"
                        writer = create_writer(chunk_path)
                        chunk_start_time = time.time()
                        print(f"  📹 New recording chunk: {chunk_path}")

            # ── Display ───────────────────────────────────────────────────────
            if args.show:
                cv2.imshow("Warehouse Gate Monitor  [ESC=stop]", frame)
                if cv2.waitKey(1) == 27:
                    print("\n  Stopped by user.")
                    break
    finally:
        # ═════════════════════════════════════════════════════════════════════
        #  CLEANUP
        # ═════════════════════════════════════════════════════════════════════
        # Ensure TEARDOWN packet is sent cleanly even if script is interrupted
        source.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Save data
    if event_logger:
        if not args.no_output:
            event_logger.export_csv(args.csv)
            event_logger.export_daily_attendance_csv(DEFAULT_DAILY_ATTENDANCE_CSV)
        summary = event_logger.summary()
        event_logger.close()
    else:
        summary = {
            "total_exits": total_exits,
            "total_returned": total_entries,
            "currently_outside": zone.outside_count,
            "avg_duration_secs": 0.0,
        }

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total exits      : {summary.get('total_exits', total_exits)}")
    print(f"  Total returns    : {summary.get('total_returned', total_entries)}")
    print(f"  Still outside    : {summary.get('currently_outside', zone.outside_count)}")
    print(f"  Overstay alerts  : {alert_count}")
    print(f"  Avg duration     : {fmt_duration(summary.get('avg_duration_secs', 0))}")
    print(f"  Duration         : {fmt_duration(elapsed_total)}")
    print(f"  Database         : {args.db}")
    print(f"{'='*60}\n")



def main():
    """Entry point — dispatches to the correct mode."""
    args = parse_args()

    if args.mode == "sources-check":
        # Camera diagnostics mode
        from camera_manager import run_sources_check
        run_sources_check(
            yaml_path=args.cameras_yaml,
            duration=args.sources_check_duration,
            resize_height=args.resize,
        )

    elif args.mode == "vehicles":
        # Vehicle ANPR mode
        main_vehicles(args)

    elif args.mode == "search":
        # Person search mode — full pipeline
        main_search(args)

    elif args.mode == "attendance":
        # Original attendance mode — call existing logic
        main_attendance()

    else:
        print(f"  Unknown mode: {args.mode}")
        sys.exit(1)


def main_vehicles(args):
    """Vehicle ANPR mode — detect vehicles, read plates, log events."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*60}")
    print(f"  VEHICLE ANPR MODE")
    print(f"{'='*60}")

    # Determine source
    source = getattr(args, 'source', None)
    if not source:
        camera_id = getattr(args, 'camera_id', None)
        if camera_id:
            import yaml
            yaml_path = getattr(args, 'cameras_yaml', 'cameras.yaml')
            with open(yaml_path) as f:
                cams = yaml.safe_load(f) or {}
            cam = cams.get(camera_id, {})
            source = cam.get('hires_stream') or cam.get('url', '')
            print(f"  Camera '{camera_id}': {source}")
        else:
            print("  ERROR: --source or --camera-id required for vehicles mode")
            sys.exit(1)

    print(f"  Source: {source}")
    print(f"  Require GPU: {getattr(args, 'require_gpu', False)}")

    # Initialize event store
    from event_store import EventStore
    store = EventStore(
        db_path=os.path.join("data", "events.db"),
        crop_dir=os.path.join("data", "crops"),
    )
    print(f"  Event store: data/events.db (WAL mode)")

    # Initialize dashboard
    from dashboard import Dashboard, broadcast_event
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 8000)

    # Initialize pipeline
    from vehicle_pipeline import VehiclePipeline
    pipeline = VehiclePipeline(
        source=source,
        frame_skip=2,
        conf_threshold=getattr(args, 'conf', 0.35),
        require_gpu=getattr(args, 'require_gpu', False),
        gpu_temp_limit=80,
        min_plate_px=60,
    )

    # Register event callback: store to DB + broadcast via SSE
    def on_track_event(event):
        event_id = store.insert_event(event)
        broadcast_event({
            "id": event_id,
            "track_id": event.track_id,
            "plate_text": event.plate_text,
            "plate_confidence": round(event.plate_confidence, 2),
            "plate_valid": event.plate_valid,
            "vehicle_type": event.vehicle_type,
            "state_code": event.state_code,
            "needs_review": event.needs_review,
            "duration_s": round(event.duration_s, 1),
            "n_plate_reads": event.n_plate_reads,
        })

    pipeline.on_event(on_track_event)

    # Start dashboard (non-blocking)
    dashboard = Dashboard(store, pipeline, host=host, port=port)
    dashboard.start(block=False)
    print(f"  Dashboard: http://{host}:{port}")

    # Check disk space
    import psutil
    disk = psutil.disk_usage(os.getcwd())
    free_gb = disk.free / (1024**3)
    if free_gb < 20:
        print(f"  ⚠ LOW DISK: {free_gb:.1f} GB free (< 20 GB threshold)")

    print(f"{'='*60}\n")
    print(f"  Pipeline starting... (Ctrl+C to stop)\n")

    # Run pipeline (blocks until stopped)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        stats = pipeline.stats
        store.close()
        print(f"\n{'='*60}")
        print(f"  SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"  Frames processed : {stats.get('frames_processed', 0)}")
        print(f"  Duration         : {stats.get('elapsed_s', 0):.0f}s")
        print(f"  Avg FPS          : {stats.get('fps', 0):.1f}")
        print(f"  Total tracks     : {stats.get('total_tracks', 0)}")
        print(f"  Peak VRAM        : {stats.get('vram_peak_mb', 0):.0f} MB")
        print(f"  Peak RSS         : {stats.get('rss_mb', 0):.0f} MB")
        print(f"{'='*60}\n")


def main_search(args):
    """Person search mode — 'find this person' pipeline."""
    import signal

    print(f"\n{'='*60}")
    print(f"  PERSON SEARCH MODE")
    print(f"{'='*60}")

    # ── 1. Load cameras ──────────────────────────────────────────────
    from camera_manager import CameraManager
    cam_mgr = CameraManager(args.cameras_yaml, resize_height=args.resize)

    enabled = list(cam_mgr.enabled_cameras.keys())
    print(f"  Cameras: {', '.join(enabled)}")

    # ── 2. Load face embedder ────────────────────────────────────────
    from face_embedder import InsightFaceEmbedder
    embedder = InsightFaceEmbedder(det_size=(320, 320))

    # ── 3. Initialize observation store ──────────────────────────────
    from observation_store import ObservationStore
    store = ObservationStore()
    print(f"  Observation store: {store.observation_count} existing observations")

    # ── 4. Start ingest pipeline ─────────────────────────────────────
    from ingest_pipeline import IngestManager

    # Detector args passed to IngestManager; it creates one PersonDetector
    # per camera so each has its own YOLO tracker state.
    detector_kwargs = dict(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
    )

    cam_mgr.start_all()
    ingest = IngestManager(cam_mgr, embedder, store,
                            detector_kwargs=detector_kwargs,
                            detect_fps=5.0, idle_fps=1.0)
    ingest.start_all()

    print(f"  Ingest pipeline running on {len(ingest.workers)} cameras")

    # ── 5. Handle search command ─────────────────────────────────────
    photo = getattr(args, 'photo', None)
    watch = getattr(args, 'watch', False)
    web = getattr(args, 'web', False)

    if web:
        # Start web UI
        from search_api import start_web_server
        start_web_server(store, embedder, cam_mgr, ingest,
                         port=getattr(args, 'web_port', 8501))
        print(f"\n  Web UI available. Ingest running in background.")
        print(f"  Press Ctrl+C to stop.\n")

    if photo:
        # Direct search
        from search_api import cli_search
        cli_search(photo, store, embedder,
                   last_minutes=getattr(args, 'last_minutes', 30),
                   watch=watch, camera_ids=None)
    elif not web:
        # Interactive mode
        from search_api import interactive_search
        interactive_search(store, embedder)

    # Keep running if web mode or watch mode
    if web or watch:
        def _shutdown(sig, frame):
            print("\n  Shutting down...")
            ingest.stop_all()
            cam_mgr.stop_all()
            ingest.print_stats()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _shutdown(None, None)
    else:
        ingest.stop_all()
        cam_mgr.stop_all()
        ingest.print_stats()


if __name__ == "__main__":
    main()
