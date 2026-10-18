"""
main.py
=======
Pipeline orchestrator for the Warehouse Entry/Exit Detection System.

Composes all modules:
  config.py    → CLI args + constants + .env loading
  stream.py    → Video/RTSP source handling (with auto-reconnect)
  detector.py  → YOLOv11 person detection + tracking (with imgsz optimization)
  zone.py      → Entry/exit crossing detection (with debounce)
  logger.py    → SQLite event logging (IST timestamps)
  visualizer.py → OpenCV drawing functions
  face_id.py   → (optional) Face recognition

Flow per frame:
  1. Read frame from source (auto-resized from native to 720p)
  2. Run YOLOv11 detection + BoT-SORT tracking (at imgsz=480 for speed)
  3. For each tracked person, check zone crossing (with debounce)
  4. Log any crossing events to SQLite (IST timestamps)
  5. Draw visualization
  6. Display / record

Usage:
    # Video file:
    python main.py --source cctv.mp4

    # Live RTSP (URL loaded from .env):
    python main.py --source rtsp

    # Live RTSP with explicit URL:
    python main.py --source rtsp://CAMERA_IP/live/0/MAIN

    # With recording and custom settings:
    python main.py --source rtsp --record --line 0.5 --imgsz 384
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


def main():
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


if __name__ == "__main__":
    main()
