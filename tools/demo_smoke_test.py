"""
tools/demo_smoke_test.py
========================
Quick smoke test: run pipeline on a clip for N frames, report results.
Not a feature — crash-detection and reporting tool.
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "data/clips/_benchmark_synthetic.mp4"
    max_processed = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    print(f"\n  DEMO SMOKE TEST")
    print(f"  Source: {source}")
    print(f"  Max frames: {max_processed}\n")

    from vehicle_pipeline import VehiclePipeline
    from event_store import EventStore

    db_path = "data/_smoke_events.db"
    crop_dir = "data/_smoke_crops"

    store = EventStore(db_path, crop_dir=crop_dir)

    pipeline = VehiclePipeline(
        source=source,
        frame_skip=2,
        conf_threshold=0.35,
        require_gpu=False,
        gpu_temp_limit=80,
        min_plate_px=60,
    )

    events_received = []

    def on_event(ev):
        events_received.append(ev)
        print(f"  EVENT: track={ev.track_id} plate='{ev.plate_text}' "
              f"conf={ev.plate_confidence:.2f} valid={ev.plate_valid} "
              f"type={ev.vehicle_type} reads={ev.n_plate_reads}")

    pipeline.on_event(on_event)

    # Run limited frames
    pipeline._running = True
    pipeline._start_time = time.time()
    pipeline._cap = pipeline._open_source()

    if not pipeline._cap or not pipeline._cap.isOpened():
        print(f"  FAIL: Cannot open source: {source}")
        return

    src_w = int(pipeline._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(pipeline._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = pipeline._cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"  Source: {src_w}x{src_h} @ {src_fps:.0f} FPS\n")

    t_start = time.time()
    n_vehicles_total = 0
    n_plates_total = 0

    try:
        while pipeline._process_count < max_processed:
            ret, frame = pipeline._cap.read()
            if not ret:
                print("  End of video")
                break

            pipeline._frame_count += 1
            if pipeline._frame_count % pipeline.frame_skip != 0:
                continue
            pipeline._process_count += 1

            # Detection
            detections = pipeline._detector.detect(frame)
            n_v = 0
            if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                n_v = len(detections.xyxy)
            n_vehicles_total += n_v

            # Plate reading
            plate_reads_by_det = {}
            vehicle_types = {}
            if n_v > 0:
                for i, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        reads = pipeline._plate_reader.read_plates(crop)
                        if reads:
                            plate_reads_by_det[i] = reads
                            n_plates_total += len(reads)
                    if detections.class_id is not None and i < len(detections.class_id):
                        vehicle_types[i] = pipeline._detector.get_class_name(
                            detections.class_id[i]
                        )

            # Tracking
            active, emitted = pipeline._tracker.update(
                detections, frame,
                plate_reads_by_det=plate_reads_by_det,
                vehicle_types=vehicle_types,
            )

            for ev in emitted:
                on_event(ev)
                store.insert_event(ev)

            if pipeline._process_count % 20 == 0:
                elapsed = time.time() - t_start
                fps = pipeline._process_count / elapsed if elapsed > 0 else 0
                thermal = pipeline._detector.check_thermal()
                print(f"  Frame {pipeline._process_count}: {n_v} vehicles, "
                      f"{len(active)} tracks, {fps:.1f} FPS, "
                      f"GPU {thermal.get('gpu_temp_c', '?')}C")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  CRASH: {e}")

    finally:
        elapsed = time.time() - t_start
        fps = pipeline._process_count / elapsed if elapsed > 0 else 0
        thermal = pipeline._detector.check_thermal()

        pipeline._cap.release()
        pipeline._detector.close()

        print(f"\n  === RESULTS ===")
        print(f"  Frames processed: {pipeline._process_count}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Vehicle detections: {n_vehicles_total}")
        print(f"  Plate reads: {n_plates_total}")
        print(f"  Events emitted: {len(events_received)}")
        valid_events = [e for e in events_received if e.plate_valid]
        print(f"  Valid-format plates: {len(valid_events)}/{len(events_received)}")
        print(f"  GPU temp: {thermal.get('gpu_temp_c', '?')}C")
        print(f"  Throttled: {thermal.get('throttled', False)}")
        print(f"  VRAM: {pipeline._detector.vram_peak_mb:.0f} MB peak")

        import psutil
        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"  RSS: {rss:.0f} MB")

        store.close()

        # Cleanup
        for f in [db_path, db_path + "-wal", db_path + "-shm"]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(crop_dir):
            shutil.rmtree(crop_dir)

        print(f"\n  Status: {'PASS' if pipeline._process_count > 0 else 'FAIL'}")
        print(f"  Label: phone / development only\n")


if __name__ == "__main__":
    main()
