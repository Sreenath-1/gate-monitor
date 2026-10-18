"""
visualizer.py
=============
All OpenCV drawing functions for the gate monitor display.

Extracted from gate_monitor.py — keeps the visual design identical but
decoupled from business logic. All functions take a frame and draw onto it
in-place (mutating the array).
"""

import cv2
from datetime import datetime

from config import C_LINE, C_ENTER, C_EXIT, C_ALERT, C_TEXT, C_PANEL


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_duration(seconds):
    """Format seconds into a human-readable duration string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def fmt_time(iso_str):
    """Format an ISO timestamp string to HH:MM:SS, or '—' if None."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return str(iso_str)


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def draw_gate_line(frame, zone):
    """Draw the gate crossing line with a translucent highlight band.

    Args:
        frame: BGR numpy array (modified in-place).
        zone: CrossingZone instance (provides line endpoints and direction).
    """
    (x1, y1), (x2, y2) = zone.get_line_endpoints()
    frame_h, frame_w = frame.shape[:2]

    cv2.line(frame, (x1, y1), (x2, y2), C_LINE, 2)

    # Translucent highlight band around the line
    overlay = frame.copy()
    if zone.direction == "horizontal":
        cv2.rectangle(overlay, (0, zone.line_coord - 8),
                      (frame_w, zone.line_coord + 8), C_LINE, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.putText(frame, "GATE LINE", (frame_w - 130, zone.line_coord - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_LINE, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, (zone.line_coord - 8, 0),
                      (zone.line_coord + 8, frame_h), C_LINE, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.putText(frame, "GATE", (zone.line_coord + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_LINE, 1, cv2.LINE_AA)
        cv2.putText(frame, "LINE", (zone.line_coord + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_LINE, 1, cv2.LINE_AA)


def draw_person_box(frame, x1, y1, x2, y2, label, sublabel, color):
    """Draw a bounding box with label and optional sublabel.

    Args:
        frame: BGR numpy array (modified in-place).
        x1, y1, x2, y2: Bounding box coordinates.
        label: Primary label text (e.g., "Person #5 [OUTSIDE]").
        sublabel: Secondary label text (e.g., "2m 30s out"). Empty string = no sublabel.
        color: BGR tuple for box and text background.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

    if sublabel:
        cv2.putText(frame, sublabel, (x1 + 3, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def draw_centroid(frame, cx, cy, color):
    """Draw a small dot at the person's centroid."""
    cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_left_panel(frame, sessions, outside_track_ids, overstay_mins):
    """Draw the left-side activity log panel.

    Args:
        frame: BGR numpy array (modified in-place).
        sessions: List of session dicts from EventLogger.get_all_sessions_for_display().
        outside_track_ids: Set of track IDs currently outside.
        overstay_mins: Overstay alert threshold in minutes.
    """
    frame_h = frame.shape[0]
    pw = 260  # Panel width

    # Semi-transparent panel background
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (pw, frame_h), (8, 12, 20), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (pw, 0), (pw, frame_h), (30, 40, 55), 1)

    # Header
    cv2.putText(frame, "GATE ACTIVITY LOG", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 120, 140), 1, cv2.LINE_AA)
    cv2.line(frame, (10, 48), (pw - 10, 48), (30, 40, 55), 1)

    y = 65
    # Show last 8 sessions (most recent first — sessions list is already DESC)
    for s in sessions[:8]:
        tid = s.get("track_id", 0)
        person = s.get("person_name", f"Person #{tid}")
        is_out = tid in outside_track_ids
        returned = s.get("status") == "returned"

        # Status color and text
        if returned:
            color = C_ENTER
            dur = s.get("duration_secs")
            status = f"✓ Returned ({fmt_duration(dur)})" if dur else "✓ Returned"
        elif is_out:
            exit_time = s.get("exit_time")
            if exit_time:
                try:
                    exit_dt = datetime.fromisoformat(exit_time)
                    secs_out = (datetime.now() - exit_dt).total_seconds()
                except (ValueError, TypeError):
                    secs_out = 0
            else:
                secs_out = 0

            if secs_out >= overstay_mins * 60:
                color = C_ALERT
                status = f"⚠ {fmt_duration(secs_out)} OUT"
            else:
                color = C_EXIT
                status = f"Outside {fmt_duration(secs_out)}"
        else:
            color = (120, 120, 120)
            status = "Inside"

        cv2.putText(frame, person, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        cv2.putText(frame, status, (10, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # Exit time
        exit_str = f"Out: {fmt_time(s.get('exit_time'))}"
        cv2.putText(frame, exit_str, (10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 140), 1, cv2.LINE_AA)

        # Return time + duration
        if s.get("entry_time"):
            dur = s.get("duration_secs", 0)
            cv2.putText(frame, f"Back: {fmt_time(s['entry_time'])}  [{fmt_duration(dur)}]",
                        (10, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (80, 160, 100), 1, cv2.LINE_AA)

        cv2.line(frame, (10, y + 50), (pw - 10, y + 50), (25, 35, 48), 1)
        y += 58
        if y > frame_h - 30:
            break


def draw_top_bar(frame, total_exits, currently_outside, alert_count, fps):
    """Draw the top status bar with aggregate stats.

    Args:
        frame: BGR numpy array (modified in-place).
        total_exits: Total number of exit events.
        currently_outside: Number of people currently outside.
        alert_count: Number of overstay alerts triggered.
        fps: Current processing FPS.
    """
    frame_w = frame.shape[1]
    bh = 52  # Bar height

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, bh), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, bh), (frame_w, bh), (30, 40, 55), 1)

    # Title
    cv2.putText(frame, "WAREHOUSE GATE MONITOR", (270, 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    # Stats
    x = 270
    items = [
        (f"Exits: {total_exits}",              C_EXIT),
        (f"Outside: {currently_outside}",      C_EXIT if currently_outside == 0 else (0, 160, 255)),
        (f"Alerts: {alert_count}",             C_ENTER if alert_count == 0 else C_ALERT),
        (f"FPS: {fps:.1f}",                    (80, 100, 120)),
    ]
    for text, color in items:
        cv2.putText(frame, text, (x, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x += tw + 30


def draw_event_flash(frame, text, color):
    """Flash a large event notification in the center of the frame.

    Args:
        frame: BGR numpy array (modified in-place).
        text: Notification text.
        color: BGR tuple for the text.
    """
    frame_h, frame_w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    tx = (frame_w - tw) // 2
    ty = frame_h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 15), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
