"""
config.py
=========
Central configuration for the gate monitor pipeline.

Supports two operational modes:
  --mode attendance   Original warehouse entry/exit detection (default)
  --mode search       "Find this person" open-set face search
  --mode sources-check  Camera diagnostics (check all cameras for N seconds)

All tuneable constants, CLI argument parsing, and zone definitions live here.
Other modules import from this file — nothing is hardcoded elsewhere.

RTSP credentials and secrets are loaded from a .env file (never hardcoded).
Camera definitions (non-secret) are loaded from cameras.yaml.
"""

import argparse
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  .ENV LOADING
# ─────────────────────────────────────────────────────────────────────────────
# Manual .env parser — avoids requiring python-dotenv as a dependency.
# Reads key=value pairs from .env file in the project root.
def _load_dotenv(env_path=None):
    """Load .env file into os.environ. Skips comments and blank lines.

    Does NOT override existing environment variables — system env takes
    precedence over .env, so you can override per-session if needed.
    """
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Don't override existing env vars
            if key not in os.environ:
                os.environ[key] = value

# Load .env on import
_load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS          = "yolo11n.pt"     # YOLO11 nano — fastest, good enough for person class
PERSON_CLASS     = 0                # COCO class 0 = person
CONF_THRESHOLD   = 0.35            # Detection confidence threshold
IOU_THRESHOLD    = 0.45            # NMS IoU threshold
DEFAULT_TRACKER  = "custom_botsort.yaml"  # Custom BoT-SORT with ReID enabled

# YOLO inference resolution — 640 is the native YOLO resolution and provides best accuracy.
# With GPU acceleration, 640 runs comfortably in real-time.
DEFAULT_IMGSZ    = 640

# ─────────────────────────────────────────────────────────────────────────────
#  ZONE / CROSSING DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LINE_POS     = 0.50        # Gate line at 50% of frame width (vertical)
DEFAULT_DIRECTION    = "vertical"  # "horizontal" or "vertical"
MIN_TRACK_DURATION   = 1.5         # Seconds — crossing events ignored if track is younger than this
DEBOUNCE_FRAMES      = 3           # Consecutive frames on new side before crossing is counted
OVERSTAY_MINUTES     = 30          # Alert threshold for time outside

# ─────────────────────────────────────────────────────────────────────────────
#  STREAM / RTSP DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
# RTSP URL loaded from environment / .env — NEVER hardcoded here
DEFAULT_RTSP_URL         = os.environ.get("RTSP_URL", "rtsp://CAMERA_IP/live/0/MAIN")
RTSP_RECONNECT_DELAY     = 2       # Seconds between reconnect attempts
RTSP_MAX_RECONNECT_SECS  = 30      # Give up after this many seconds of failures
RTSP_FRAME_TIMEOUT       = 5       # Consecutive failed reads before triggering reconnect
VIDEO_CHUNK_MINUTES      = 10      # When recording a stream, rotate output files every N minutes
DEFAULT_RESIZE_HEIGHT    = 720     # Downscale high-res streams to this height for CPU inference

# ─────────────────────────────────────────────────────────────────────────────
#  TIMEZONE
# ─────────────────────────────────────────────────────────────────────────────
# All timestamps are stored in IST (UTC+5:30) explicitly.
# This avoids the silent UTC drift from SQLite's datetime('now').
import datetime as _dt
IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30), name="IST")

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING / OUTPUT DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH      = "gate_events.db"
DEFAULT_OUTPUT_VIDEO = "gate_output.mp4"
DEFAULT_CSV_EXPORT   = "gate_report.csv"
DEFAULT_DAILY_ATTENDANCE_CSV = "daily_attendance_summary.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  FACE QUALITY GATE (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────
# Lightweight checks run BEFORE face recognition — reject frames that would
# produce garbage embeddings, without wasting GPU/CPU cycles on ArcFace.
MIN_FACE_PX       = int(os.environ.get("GATE_MIN_FACE_PX", "40"))
MIN_SHARPNESS     = float(os.environ.get("GATE_MIN_SHARPNESS", "15.0"))
MAX_YAW_DEG       = int(os.environ.get("GATE_MAX_YAW_DEG", "45"))
MAX_PITCH_DEG     = int(os.environ.get("GATE_MAX_PITCH_DEG", "35"))
MIN_BRIGHTNESS    = int(os.environ.get("GATE_MIN_BRIGHTNESS", "40"))
MAX_BRIGHTNESS    = int(os.environ.get("GATE_MAX_BRIGHTNESS", "220"))

# ─────────────────────────────────────────────────────────────────────────────
#  FACE RECOGNITION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
GATE_FACE_THRESHOLD = float(os.environ.get("GATE_FACE_THRESHOLD", "0.6"))
GATE_PERIOCULAR_THRESHOLD = float(os.environ.get("GATE_PERIOCULAR_THRESHOLD", "0.3"))
GATE_PERIOCULAR_VOTE_FLOOR = float(os.environ.get("GATE_PERIOCULAR_VOTE_FLOOR", "0.15"))

# ─────────────────────────────────────────────────────────────────────────────
#  OPEN-SET REJECTION — ABSOLUTE SIMILARITY FLOORS
# ─────────────────────────────────────────────────────────────────────────────
# These floors are the SECOND criterion in dual-criterion matching.
# A candidate match must pass BOTH the calibrated NAC/gallery-relative threshold
# AND the raw absolute cosine similarity floor against the matched template.
#
# ⚠  PROVISIONAL VALUES — validated against ArcFace literature baselines:
#   Full-face 0.40: ArcFace cosine-sim for unrelated people typically < 0.25;
#                   0.40 gives generous margin above noise while rejecting strangers.
#   Periocular 0.30: periocular embeddings are noisier; 0.30 is conservative.
#   These MUST be validated against real held-out unknown-person footage before
#   treating them as production-ready. See scripts/test_open_set_rejection.py.
GATE_ABS_FLOOR_FULLFACE = float(os.environ.get("GATE_ABS_FLOOR_FULLFACE", "0.40"))
GATE_ABS_FLOOR_PERIOCULAR = float(os.environ.get("GATE_ABS_FLOOR_PERIOCULAR", "0.35"))

# ─────────────────────────────────────────────────────────────────────────────
#  THIN-MARGIN FLAGGING (open-set safety)
# ─────────────────────────────────────────────────────────────────────────────
# If the gap between the winning identity's score and the next-best enrolled
# identity's score is below this margin, downgrade to "uncertain" regardless of
# raw confidence. A match that barely beat a different enrolled person is
# ambiguous and needs human review.
GATE_THIN_MARGIN_THRESHOLD = float(os.environ.get("GATE_THIN_MARGIN", "0.05"))

# ─────────────────────────────────────────────────────────────────────────────
#  DUPLICATE ENROLLMENT GUARD
# ─────────────────────────────────────────────────────────────────────────────
# During enrollment, if a new face's embedding has cosine similarity above this
# threshold to any already-enrolled *different* identity, halt with a warning.
GATE_DUPLICATE_ENROLL_THRESHOLD = float(os.environ.get("GATE_DUPLICATE_ENROLL_THRESH", "0.50"))

# ─────────────────────────────────────────────────────────────────────────────
#  TRACK-LEVEL EMBEDDING FUSION (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────
# Per-track bounded buffer of (embedding, quality_score, match_name, similarity).
# Identity = majority vote of top-K matches, tie-broken by quality-weighted avg.
TRACK_EMBEDDING_BUFFER_K = int(os.environ.get("GATE_TRACK_BUFFER_K", "5"))

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE-TIERED DECISIONS (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────
# HIGH_CONF: auto-accept identity (no human review needed)
# LOW_CONF:  below this = "unresolved" — identity unknown
# Between them = "uncertain" — log with needs_review flag
HIGH_CONF_THRESHOLD = float(os.environ.get("GATE_HIGH_CONF", "0.75"))
LOW_CONF_THRESHOLD  = float(os.environ.get("GATE_LOW_CONF", "0.35"))

# ─────────────────────────────────────────────────────────────────────────────
#  TAILGATING DETECTION (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────
TAILGATE_IOU_THRESHOLD = float(os.environ.get("GATE_TAILGATE_IOU", "0.3"))
TAILGATE_MIN_FRAMES    = int(os.environ.get("GATE_TAILGATE_MIN_FRAMES", "5"))

# ─────────────────────────────────────────────────────────────────────────────
#  STRUCTURED DEBUG LOGGING (Stage 7)
# ─────────────────────────────────────────────────────────────────────────────
DEBUG_LOG_FILE = os.environ.get("GATE_DEBUG_LOG", "gate_debug.log")

# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY COLORS (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
C_LINE    = (0,  80, 255)     # Orange-red — gate line
C_ENTER   = (0, 220,  80)    # Green — entering / returned
C_EXIT    = (0, 120, 255)    # Orange — exiting / outside
C_ALERT   = (0,   0, 220)    # Red — overstay alert
C_TEXT    = (255, 255, 255)   # White text
C_PANEL   = (10,  10,  10)   # Near-black panel background


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
# Valid operational modes
VALID_MODES = ("attendance", "search", "sources-check", "vehicles")

# ─────────────────────────────────────────────────────────────────────────────
#  VEHICLE ANPR DEFAULTS (--mode vehicles)
# ─────────────────────────────────────────────────────────────────────────────
VEHICLE_FRAME_SKIP         = int(os.environ.get("VEHICLE_FRAME_SKIP", "2"))       # Process every Nth frame (12-15 FPS from 25 FPS)
VEHICLE_DETECT_IMGSZ       = int(os.environ.get("VEHICLE_DETECT_IMGSZ", "640"))   # Detection inference resolution
VEHICLE_CONF_THRESHOLD     = float(os.environ.get("VEHICLE_CONF_THRESHOLD", "0.35"))
VEHICLE_MIN_PLATE_PX       = int(os.environ.get("VEHICLE_MIN_PLATE_PX", "60"))    # Minimum plate width in pixels
VEHICLE_PLATE_PAD_RATIO    = float(os.environ.get("VEHICLE_PLATE_PAD", "0.12"))   # Pad plate box by 12%
VEHICLE_QUALITY_TOPK       = int(os.environ.get("VEHICLE_QUALITY_TOPK", "8"))     # Best crops per track
VEHICLE_EMIT_THRESHOLD     = float(os.environ.get("VEHICLE_EMIT_THRESHOLD", "0.5"))
VEHICLE_EMIT_MIN_READS     = int(os.environ.get("VEHICLE_EMIT_MIN_READS", "3"))
VEHICLE_MERGE_DISTANCE     = int(os.environ.get("VEHICLE_MERGE_DIST", "1"))       # Edit distance for track merge
VEHICLE_MERGE_WINDOW_SECS  = int(os.environ.get("VEHICLE_MERGE_WINDOW", "60"))    # Seconds for track merge window
VEHICLE_CROP_RETENTION_DAYS = int(os.environ.get("VEHICLE_CROP_RETENTION", "30"))
VEHICLE_GPU_TEMP_LIMIT     = int(os.environ.get("VEHICLE_GPU_TEMP_LIMIT", "80"))  # Adaptive load-shedding threshold



def build_argparser():
    """Build and return the argument parser (does not call parse_args yet)."""
    p = argparse.ArgumentParser(
        description="Gate Monitor — Attendance Tracker & Person Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  attendance     Original warehouse entry/exit detection (default)
  search         "Find this person" open-set face search across cameras
  sources-check  Camera diagnostic — check all cameras for 30s and report

Examples:
  # Attendance mode (default — unchanged behavior):
  python main.py --source cctv.mp4
  python main.py --source rtsp

  # Check all cameras defined in cameras.yaml:
  python main.py --mode sources-check

  # Person search mode (Stages 2-5):
  python main.py --mode search
        """
    )

    # ── Mode ─────────────────────────────────────────────────────────────
    p.add_argument("--mode", type=str, default="attendance",
                   choices=VALID_MODES,
                   help="Operational mode. Default: attendance")

    # ── Camera config ────────────────────────────────────────────────────
    p.add_argument("--cameras-yaml", type=str,
                   default=str(Path(__file__).parent / "cameras.yaml"),
                   help="Path to cameras.yaml config. Default: cameras.yaml")
    p.add_argument("--sources-check-duration", type=int, default=30,
                   help="Duration in seconds for sources-check mode. Default: 30")

    # ── Search mode options ──────────────────────────────────────────────
    p.add_argument("--photo", type=str, default=None,
                   help="Path to target photo for search mode.")
    p.add_argument("--last-minutes", type=float, default=30,
                   help="Search the last N minutes. Default: 30")
    p.add_argument("--face-index", type=int, default=None,
                   help="If multiple faces in photo, select this index (0-based).")
    p.add_argument("--watch", action="store_true",
                   help="After retro search, keep watching for live matches.")
    p.add_argument("--web", action="store_true",
                   help="Start FastAPI web UI for search (127.0.0.1:8501).")
    p.add_argument("--web-port", type=int, default=8501,
                   help="Port for web UI. Default: 8501")

    # ── Source (for attendance mode / single-camera) ─────────────────────
    p.add_argument("--source", type=str, default="cctv.mp4",
                   help="Video file path, RTSP URL, or 'rtsp' to load URL from .env. "
                        "Used by attendance mode. Default: cctv.mp4")

    # ── Detection ────────────────────────────────────────────────────────
    p.add_argument("--weights", type=str, default=WEIGHTS,
                   help=f"YOLO model weights. Default: {WEIGHTS}")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help=f"Detection confidence threshold. Default: {CONF_THRESHOLD}")
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                   help=f"NMS IoU threshold. Default: {IOU_THRESHOLD}")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help=f"YOLO internal inference resolution. Smaller = faster. "
                        f"Default: {DEFAULT_IMGSZ}")

    # ── Tracking ─────────────────────────────────────────────────────────
    p.add_argument("--tracker", type=str, default=DEFAULT_TRACKER,
                   choices=["botsort.yaml", "bytetrack.yaml"],
                   help=f"Tracker config. Default: {DEFAULT_TRACKER}")

    # ── Zone / crossing ──────────────────────────────────────────────────
    p.add_argument("--line", type=float, default=DEFAULT_LINE_POS,
                   help=f"Gate line position as fraction of frame height/width (0.0–1.0). "
                        f"Default: {DEFAULT_LINE_POS}")
    p.add_argument("--direction", type=str, default=DEFAULT_DIRECTION,
                   choices=["horizontal", "vertical"],
                   help=f"Gate line orientation. Default: {DEFAULT_DIRECTION}")
    p.add_argument("--flip", action="store_true",
                   help="Flip entry/exit direction (swap which side is 'inside' vs 'outside')")
    p.add_argument("--min-track-duration", type=float, default=MIN_TRACK_DURATION,
                   help=f"Minimum track age (seconds) before crossing events are counted. "
                        f"Filters flicker. Default: {MIN_TRACK_DURATION}")
    p.add_argument("--debounce-frames", type=int, default=DEBOUNCE_FRAMES,
                   help=f"Consecutive frames on new side before crossing is counted. "
                        f"Default: {DEBOUNCE_FRAMES}")
    p.add_argument("--overstay", type=int, default=OVERSTAY_MINUTES,
                   help=f"Overstay alert threshold in minutes. Default: {OVERSTAY_MINUTES}")

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--db", type=str, default=DEFAULT_DB_PATH,
                   help=f"SQLite database path. Default: {DEFAULT_DB_PATH}")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_VIDEO,
                   help=f"Output video filename. Default: {DEFAULT_OUTPUT_VIDEO}")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV_EXPORT,
                   help=f"CSV export filename. Default: {DEFAULT_CSV_EXPORT}")

    # ── Display ──────────────────────────────────────────────────────────
    p.add_argument("--show", action="store_true", default=True,
                   help="Show live preview window (default: True)")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Disable live preview window (faster)")

    # ── Stream-specific ──────────────────────────────────────────────────
    p.add_argument("--record", action="store_true", default=False,
                   help="Record output video for live streams (writes in timed chunks)")
    p.add_argument("--no-output", action="store_true", default=False,
                   help="Disable all file output — pure monitoring mode")
    p.add_argument("--resize", type=int, default=DEFAULT_RESIZE_HEIGHT,
                   help=f"Resize frame height for processing (0=no resize). "
                        f"Default: {DEFAULT_RESIZE_HEIGHT}")

    # ── Face recognition (optional, loaded only when dir exists) ─────────
    p.add_argument("--faces-dir", type=str,
                   default=str(Path(__file__).parent / "known_faces"),
                   help="Path to known_faces/ directory with <Name>/ subfolders of photos")
    p.add_argument("--face-threshold", type=float, default=GATE_FACE_THRESHOLD,
                   help=f"Face match confidence threshold (cosine similarity). Default: {GATE_FACE_THRESHOLD}")
    p.add_argument("--peri-threshold", type=float, default=GATE_PERIOCULAR_THRESHOLD,
                   help=f"Periocular match confidence threshold. Default: {GATE_PERIOCULAR_THRESHOLD}")
    p.add_argument("--face-every", type=int, default=5,
                   help="Run face recognition every N frames per track. Default: 5")

    # ── Stage 7: Quality gate & debug logging ─────────────────────────────
    p.add_argument("--quality-gate", action="store_true", default=True,
                   help="Enable face quality gate (default: enabled)")
    p.add_argument("--no-quality-gate", dest="quality_gate", action="store_false",
                   help="Disable face quality gate")
    p.add_argument("--debug-log", type=str, default=DEBUG_LOG_FILE,
                   help=f"Path for structured debug log. Default: {DEBUG_LOG_FILE}")

    # ── Vehicle ANPR (--mode vehicles) ──────────────────────────────────
    p.add_argument("--require-gpu", action="store_true", default=False,
                   help="Fail fast if CUDA is not the active ONNX/torch provider")
    p.add_argument("--camera-id", type=str, default="cctv_gate",
                   help="Camera ID from cameras.yaml (vehicles mode). Default: cctv_gate")
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Dashboard bind address. Default: 127.0.0.1")
    p.add_argument("--port", type=int, default=8502,
                   help="Dashboard port (vehicles mode). Default: 8502")

    return p


def parse_args():
    """Parse command-line arguments and return the namespace.

    Special handling:
      --source rtsp  → expands to the RTSP_URL from .env / environment
      --mode sources-check → routes to camera diagnostics
    """
    args = build_argparser().parse_args()

    # Shorthand: --source rtsp → load URL from .env
    if args.source.lower() == "rtsp":
        args.source = DEFAULT_RTSP_URL
        print(f"  [config] RTSP URL loaded from .env: {args.source}")

    return args
