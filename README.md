# 🚪 Employee Gate Activity Monitor

Tracks employee entry/exit through a CCTV-monitored gate.
Works with **any video file** or **live RTSP camera stream**.

---

## What It Does

| Detection | Detail |
|---|---|
| **Exit detected** | Person crosses line going outward → logged with timestamp |
| **Return detected** | Same person crosses back → duration calculated automatically |
| **Duration tracking** | Live timer shown on each person while they're outside |
| **Overstay alert** | Console + on-screen alert if outside beyond threshold (default: 30 min) |
| **Face recognition** | InsightFace ArcFace identifies known employees (with mask-robust fallback) |
| **CSV report** | Full activity log saved after every session |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run on your video
python gate_monitor.py --source cctv.mp4

# 3. Run on live RTSP camera
python gate_monitor.py --source rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
```

Output video saved as `gate_output.mp4`
CSV report saved as `gate_report.csv`

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| **Person Detection** | YOLO11n (Ultralytics) | 5.4MB nano model, ~1.8 FPS on CPU |
| **Person Tracking** | ByteTrack | Built into Ultralytics |
| **Face Recognition** | InsightFace buffalo_l (ArcFace) | 512-dim embeddings, ~99% frontal accuracy |
| **Mask Fallback** | Periocular matching | Upper-face embeddings when mask detected |
| **Inference** | CPU (ONNX Runtime) | No GPU required |

---

## All Options

```
--source       Video file path, RTSP URL, or webcam index (0)
--weights      YOLO weights (default: yolo11n.pt, auto-downloaded)
--line         Gate line position (0.0–1.0, default: 0.45)
               0.0 = top of frame, 1.0 = bottom of frame
--overstay     Alert threshold in minutes (default: 30)
--flip         Flip entry/exit direction if camera is oriented differently
--direction    Gate line orientation: horizontal (default) or vertical
--output       Output video filename (default: gate_output.mp4)
--report       CSV report filename (default: gate_report.csv)
--no-show      Disable live preview window (runs faster)
--record       Enable video recording for live RTSP streams
--resize       Frame height for processing (default: 720, 0=no resize)

Face Recognition:
--faces-dir    Path to known_faces/ directory (default: ./known_faces/)
--face-threshold  Match confidence threshold (default: 0.4 for ArcFace)
--face-every   Run face recognition every N frames (default: 15)
```

---

## Face Recognition Setup

### Adding Known Faces

Place reference photos in the `known_faces/` directory:

```
known_faces/
    Alice/
        photo1.jpg
        photo2.jpg
    OtherPerson/
        photo1.jpg
```

- Use clear, frontal-ish photos (not extreme angles)
- Multiple photos per person improve accuracy
- Both full-face and periocular (upper-face) embeddings are built automatically

### Mask-Robust Recognition

When a person wears a mask, the system automatically:
1. Detects mask via landmark analysis
2. Falls back to periocular (eyes + forehead) matching
3. Uses a stricter threshold to reduce false positives

> **Accuracy note**: Periocular-only matching is less accurate than full-face
> (confidence drops ~0.15-0.30). This is a fundamental limitation — with only
> eyes visible, different people can look more similar.

---

## Setting the Gate Line

The gate line determines where entry/exit is detected.
Run with default settings first and watch where people cross.

```bash
# Move line lower (60% down the frame):
python gate_monitor.py --source cctv.mp4 --line 0.6

# Move line higher (30% down the frame):
python gate_monitor.py --source cctv.mp4 --line 0.3
```

The line should cut across the path people walk through.

---

## Direction Logic

By default:
- Person moving **downward** (toward bottom of frame) = **EXIT**
- Person moving **upward** (toward top of frame) = **ENTRY / RETURN**

If your camera is mounted differently and exits/entries are reversed:
```bash
python gate_monitor.py --source cctv.mp4 --flip
```

---

## Live RTSP Camera

```bash
python gate_monitor.py \
  --source rtsp://USER:PASSWORD@CAMERA_IP:554/stream1 \
  --line 0.5 \
  --overstay 30 \
  --record
```

For multiple cameras, run one instance per camera in separate terminals.

---

## CSV Report Format

```
Person,Exit Time,Return Time,Duration,Status
Alice,14:22:05,14:45:30,23m 25s,Returned
Person #7,15:10:12,—,—,Still outside
```

---

## Performance Notes

| Scenario | FPS (GPU) | Notes |
|---|---|---|
| YOLO11n + InsightFace + Quality Gate | ~22-25 | RTX 4050 GPU inference |
| With --no-show | +5-10% | No display overhead |
| With --resize 480 | ~25+ | Lower resolution, faster processing |
| With --no-quality-gate | ~23-25 | Skips quality checks (not recommended) |

**Mitigation for low FPS**: Face recognition runs every `--face-every N` frames
(default: 15), not every frame. Increase this value for faster processing at
the cost of slower face identification.

---

## Stage 7: Production Features

### Face Quality Gate
Lightweight CPU checks run **before** face recognition to reject frames that
would produce garbage embeddings:
- **Face size**: Rejects tiny distant faces (< 40px minimum dimension)
- **Sharpness**: Variance of Laplacian detects motion blur
- **Pose angle**: PnP solve from 5-point landmarks rejects extreme profile views
- **Brightness**: Rejects too-dark or blown-out face crops

Enable/disable via `--quality-gate` / `--no-quality-gate` (default: ON).

### Track-Level Embedding Fusion
Instead of relying on a single frame's recognition result, each track
accumulates up to K recognition observations. Identity is decided by
majority vote, tie-broken by quality-weighted average embedding. This
stabilizes identity across flickering matches.

### Confidence-Tiered Decisions
At crossing time, each person gets a decision tier:
- **high** (confidence ≥ 0.75): Auto-accepted, no review needed
- **uncertain** (0.35 ≤ confidence < 0.75): Logged with `needs_review=True`
- **unresolved** (confidence < 0.35): Logged as "Unknown"

### Tailgating Detection
Detects when two people cross the gate line simultaneously with overlapping
bounding boxes (IoU ≥ 0.3 for ≥ 5 consecutive frames). Flagged in both
the `crossing_events` table and structured logs.

### Decision Audit Log
Every crossing writes a full audit record to the `decision_audit` table:
- Frames considered vs. frames that passed quality gate
- Rejection reason histogram (e.g., `{"TOO_BLURRY": 3, "PASS": 7}`)
- Final confidence score and decision tier
- Tailgate flag

### Structured Debug Logging
All per-frame events are logged to `gate_debug.log` (rotating, 5MB, 3 backups):
```
2026-09-22 12:30:15.123 | DEBUG | quality_gate | track=5 | score=0.234 | reason=TOO_BLURRY
2026-09-22 12:30:15.200 | DEBUG | recognition | track=5 | name=Alice | similarity=0.823 | method=full
2026-09-22 12:30:16.100 | INFO  | crossing | track=5 | event=exit | name=Alice | decision_tier=high
```

### Camera Motion Compensation (CMC)
BoT-SORT uses sparse optical flow (`sparseOptFlow`) to compensate for camera
vibration, improving track continuity without additional GPU cost (~1-3ms/frame).

---

## Environment Variable Overrides

All Stage 7 thresholds are configurable via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `GATE_MIN_FACE_PX` | 40 | Min face dimension (px) for recognition |
| `GATE_MIN_SHARPNESS` | 15.0 | Laplacian variance threshold |
| `GATE_MAX_YAW_DEG` | 45 | Max yaw angle before rejection |
| `GATE_MAX_PITCH_DEG` | 35 | Max pitch angle before rejection |
| `GATE_MIN_BRIGHTNESS` | 40 | Min mean pixel intensity |
| `GATE_MAX_BRIGHTNESS` | 220 | Max mean pixel intensity |
| `GATE_TRACK_BUFFER_K` | 5 | Embedding buffer size per track |
| `GATE_HIGH_CONF` | 0.75 | Auto-accept confidence threshold |
| `GATE_LOW_CONF` | 0.35 | Below this = unresolved |
| `GATE_TAILGATE_IOU` | 0.3 | IoU threshold for tailgate detection |
| `GATE_TAILGATE_MIN_FRAMES` | 5 | Consecutive frames for tailgate |
| `GATE_DEBUG_LOG` | gate_debug.log | Debug log file path |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Nobody detected | Lower `--conf 0.25` |
| Wrong direction | Add `--flip` flag |
| Count at wrong place | Adjust `--line` value |
| Slow on CPU | Add `--no-show` and/or `--resize 480` |
| RTSP not connecting | Test URL in VLC first |
| Face not recognized | Add more reference photos, lower `--face-threshold` |
| No face in top-down view | Face recognition requires face-visible angle |
| Too many quality rejections | Lower `GATE_MIN_SHARPNESS` or `GATE_MIN_FACE_PX` |
| False tailgate alerts | Raise `GATE_TAILGATE_IOU` (e.g., 0.5) |
| Debug log too large | Set `GATE_DEBUG_LOG` to `/dev/null` or NUL |
