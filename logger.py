"""
logger.py
=========
SQLite-based event logging for entry/exit detection.

Why SQLite over CSV:
  - Queryable: "show all entries > 30 min today" is one SQL query
  - Concurrent-safe: multiple processes can read while one writes
  - Schema-enforced: data types, constraints, no malformed rows
  - Lightweight: stdlib sqlite3, no external dependency
  - Still exportable to CSV when needed

Timestamps:
  All timestamps are stored as ISO 8601 strings in IST (UTC+05:30).
  SQLite's default datetime('now') returns UTC, which would silently be
  5.5 hours off from local time. We avoid this by always using Python's
  datetime.now(IST) and storing the result explicitly.

Two tables:
  - crossing_events: raw directional crossing log (every crossing)
  - sessions: paired exit→entry with duration (what you query for dwell time)

Track ID limitation (stated explicitly per requirement):
  Sessions are keyed on track_id, which resets if the tracker (BoT-SORT)
  loses a person (occlusion, leaving frame, re-entering later). This means
  the system measures PER-VISIT duration, NOT cumulative daily duration for
  a named person, unless sessions are re-linked via face recognition.
  Face recognition can assign a person_name, but multiple sessions for the
  same person_name are NOT automatically merged — they remain separate visits.
"""

import csv
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

from config import IST


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _now_ist() -> datetime:
    """Return current time in IST (UTC+05:30), timezone-aware."""
    return datetime.now(IST)


def _to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string (includes timezone offset)."""
    return dt.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crossing_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id      INTEGER NOT NULL,
    person_name   TEXT,
    event_type    TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
    timestamp     TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    needs_review  BOOLEAN DEFAULT 0,
    possible_tailgate BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id      INTEGER NOT NULL,
    person_name   TEXT,
    exit_time     TEXT NOT NULL,
    entry_time    TEXT,
    duration_secs REAL,
    status        TEXT DEFAULT 'outside' CHECK(status IN ('outside', 'returned', 'lost')),
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_audit (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    crossing_event_id         INTEGER REFERENCES crossing_events(id),
    track_id                  INTEGER NOT NULL,
    frames_considered         INTEGER NOT NULL DEFAULT 0,
    frames_passed_quality     INTEGER NOT NULL DEFAULT 0,
    rejection_reason_histogram TEXT,
    final_confidence          REAL,
    decision_tier             TEXT CHECK(decision_tier IN ('high', 'uncertain', 'unresolved')),
    possible_tailgate         BOOLEAN DEFAULT 0,
    created_at                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_track ON sessions(track_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_events_track ON crossing_events(track_id);
CREATE INDEX IF NOT EXISTS idx_audit_event ON decision_audit(crossing_event_id);
"""


class EventLogger:
    """SQLite event logger for entry/exit tracking.

    All timestamps stored in IST (UTC+05:30) as ISO 8601 strings.

    Usage:
        logger = EventLogger("gate_events.db")

        # Log an exit
        session_id = logger.log_exit(track_id=5, person_name="Alice")

        # Later, log the return
        logger.log_entry(track_id=5, person_name="Alice", session_id=session_id)

        # Query
        sessions = logger.get_sessions(status="outside")
        logger.export_csv("report.csv")

        logger.close()
    """

    def __init__(self, db_path: str):
        """Open (or create) the SQLite database and ensure schema exists.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
        self.conn.executescript(SCHEMA_SQL)
        self._migrate_schema()  # Stage 7: add new columns to existing DBs
        self.conn.commit()

        # In-memory map: track_id → session_id (for pairing exit with entry)
        self._active_sessions: Dict[int, int] = {}
        # In-memory map: person_name → session_id (for pairing across ID switches)
        self._active_sessions_by_name: Dict[str, int] = {}

    def _migrate_schema(self):
        """Safely add new columns to existing databases (Stage 7 migration).

        Uses ALTER TABLE which is a no-op if the column already exists
        (caught by the try/except). This allows upgrading from Stage 6 DBs.
        """
        migrations = [
            "ALTER TABLE crossing_events ADD COLUMN needs_review BOOLEAN DEFAULT 0",
            "ALTER TABLE crossing_events ADD COLUMN possible_tailgate BOOLEAN DEFAULT 0",
        ]
        for sql in migrations:
            try:
                self.conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists

    def log_exit(self, track_id: int, person_name: Optional[str] = None,
                 timestamp: Optional[datetime] = None,
                 needs_review: bool = False,
                 possible_tailgate: bool = False) -> int:
        """Log an exit crossing event and create a new session.

        Args:
            track_id: Tracking ID from the detector.
            person_name: Identified person name, or None for anonymous.
            timestamp: Event time in IST. Defaults to now (IST).
            needs_review: Stage 7 — True if identity confidence is uncertain.
            possible_tailgate: Stage 7 — True if tailgating detected.

        Returns:
            session_id: The ID of the newly created session row.
        """
        ts = timestamp or _now_ist()
        ts_str = _to_iso(ts)
        display = person_name or f"Person #{track_id}"
        created = _to_iso(_now_ist())

        # Insert raw crossing event
        cursor = self.conn.execute(
            "INSERT INTO crossing_events "
            "(track_id, person_name, event_type, timestamp, created_at, "
            " needs_review, possible_tailgate) "
            "VALUES (?, ?, 'exit', ?, ?, ?, ?)",
            (track_id, display, ts_str, created,
             1 if needs_review else 0, 1 if possible_tailgate else 0)
        )
        crossing_event_id = cursor.lastrowid

        # Create session (starts with exit, no return yet)
        cursor = self.conn.execute(
            "INSERT INTO sessions (track_id, person_name, exit_time, status, created_at) "
            "VALUES (?, ?, ?, 'outside', ?)",
            (track_id, display, ts_str, created)
        )
        session_id = cursor.lastrowid
        self.conn.commit()

        self._active_sessions[track_id] = session_id
        if person_name:
            self._active_sessions_by_name[person_name] = session_id
        return session_id

    def log_entry(self, track_id: int, person_name: Optional[str] = None,
                  session_id: Optional[int] = None,
                  timestamp: Optional[datetime] = None,
                  needs_review: bool = False,
                  possible_tailgate: bool = False) -> Optional[float]:
        """Log an entry (return) crossing event and close the session.

        Args:
            track_id: Tracking ID from the detector.
            person_name: Identified person name, or None.
            session_id: Explicit session ID to close. If None, uses the
                        most recent active session for this track_id.
            timestamp: Event time in IST. Defaults to now (IST).
            needs_review: Stage 7 — True if identity confidence is uncertain.
            possible_tailgate: Stage 7 — True if tailgating detected.

        Returns:
            duration_secs: Duration the person was outside, or None if no matching session.
        """
        ts = timestamp or _now_ist()
        ts_str = _to_iso(ts)
        display = person_name or f"Person #{track_id}"
        created = _to_iso(_now_ist())

        # Insert raw crossing event
        self.conn.execute(
            "INSERT INTO crossing_events "
            "(track_id, person_name, event_type, timestamp, created_at, "
            " needs_review, possible_tailgate) "
            "VALUES (?, ?, 'entry', ?, ?, ?, ?)",
            (track_id, display, ts_str, created,
             1 if needs_review else 0, 1 if possible_tailgate else 0)
        )

        # Find the session to close
        sid = session_id
        if sid is None and person_name:
            # 1. Try to link by person_name (handles cumulative visits/ID switches)
            sid = self._active_sessions_by_name.get(person_name)
        if sid is None:
            # 2. Fallback to track_id (for unrecognized people)
            sid = self._active_sessions.get(track_id)

        duration = None

        if sid is not None:
            row = self.conn.execute(
                "SELECT exit_time, track_id, person_name FROM sessions WHERE id = ?", (sid,)
            ).fetchone()

            if row:
                exit_time = datetime.fromisoformat(row["exit_time"])
                duration = (ts - exit_time).total_seconds()

                self.conn.execute(
                    "UPDATE sessions SET entry_time = ?, duration_secs = ?, "
                    "person_name = ?, status = 'returned' WHERE id = ?",
                    (ts_str, duration, display, sid)
                )

                # Clean up memory caches
                orig_track_id = row["track_id"]
                orig_name = row["person_name"]
                self._active_sessions.pop(orig_track_id, None)
                if orig_name:
                    self._active_sessions_by_name.pop(orig_name, None)
                
                # Also clean up current tracking info just in case
                self._active_sessions.pop(track_id, None)
                if person_name:
                    self._active_sessions_by_name.pop(person_name, None)

        self.conn.commit()
        return duration

    def update_person_name(self, track_id: int, person_name: str):
        """Update the person name for an active session (e.g., after face recognition matches).

        Args:
            track_id: Tracking ID.
            person_name: Newly identified name.
        """
        sid = self._active_sessions.get(track_id)
        if sid is not None:
            self.conn.execute(
                "UPDATE sessions SET person_name = ? WHERE id = ?",
                (person_name, sid)
            )
            self.conn.commit()
            self._active_sessions_by_name[person_name] = sid

    def close_session(self, track_id: int, status: str = "lost", timestamp: Optional[datetime] = None):
        """Close an open session due to track split or loss, stopping the overstay timer."""
        sid = self._active_sessions.get(track_id)
        if sid is not None:
            ts = timestamp or _now_ist()
            ts_str = _to_iso(ts)
            
            row = self.conn.execute(
                "SELECT exit_time, person_name FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            
            if row:
                exit_time = datetime.fromisoformat(row["exit_time"])
                duration = (ts - exit_time).total_seconds()
                
                self.conn.execute(
                    "UPDATE sessions SET entry_time = ?, duration_secs = ?, status = ? WHERE id = ?",
                    (ts_str, duration, status, sid)
                )
                
                orig_name = row["person_name"]
                self._active_sessions.pop(track_id, None)
                if orig_name:
                    self._active_sessions_by_name.pop(orig_name, None)
            
            self.conn.commit()

    def get_active_session(self, track_id: int) -> Optional[Dict]:
        """Get the active (open) session for a track, if any."""
        sid = self._active_sessions.get(track_id)
        if sid is None:
            return None
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        return dict(row) if row else None

    def get_sessions(self, status: Optional[str] = None,
                     limit: int = 50) -> List[Dict]:
        """Query sessions, optionally filtered by status.

        Args:
            status: Filter by "outside", "returned", or "lost". None = all.
            limit: Max rows to return.

        Returns:
            List of session dicts.
        """
        if status:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_sessions_for_display(self) -> List[Dict]:
        """Get all sessions ordered by most recent first, for the UI panel."""
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_exit_time(self, track_id: int) -> Optional[datetime]:
        """Get the exit time for an active session."""
        sid = self._active_sessions.get(track_id)
        if sid is None:
            return None
        row = self.conn.execute(
            "SELECT exit_time FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        if row:
            return datetime.fromisoformat(row["exit_time"])
        return None

    def export_csv(self, csv_path: str):
        """Export all sessions to a CSV file (backwards-compatible format).

        Args:
            csv_path: Output CSV file path.
        """
        sessions = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id ASC"
        ).fetchall()

        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Gate Activity Report"])
            writer.writerow(["Generated", _to_iso(_now_ist())])
            writer.writerow(["Timezone", "IST (UTC+05:30)"])
            writer.writerow([])
            writer.writerow(["Person", "Exit Time", "Entry Time", "Duration (s)", "Status"])

            for s in sessions:
                dur = f"{s['duration_secs']:.1f}" if s['duration_secs'] is not None else "—"
                entry = s['entry_time'] or "—"
                writer.writerow([
                    s['person_name'] or f"Person #{s['track_id']}",
                    s['exit_time'],
                    entry,
                    dur,
                    s['status'],
                ])

        print(f"  📄 CSV report exported: {csv_path}")

    def summary(self) -> Dict:
        """Return aggregate statistics."""
        total_exits = self.conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        total_returned = self.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'returned'"
        ).fetchone()[0]
        currently_outside = self.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'outside'"
        ).fetchone()[0]
        avg_duration = self.conn.execute(
            "SELECT AVG(duration_secs) FROM sessions WHERE duration_secs IS NOT NULL AND status = 'returned'"
        ).fetchone()[0]

        return {
            "total_exits": total_exits,
            "total_returned": total_returned,
            "currently_outside": currently_outside,
            "avg_duration_secs": avg_duration or 0.0,
        }

    def export_daily_attendance_csv(self, csv_path: str):
        """Export a daily attendance summary CSV aggregated per person per day.

        Columns:
          - Person (employee name or ID)
          - Date (YYYY-MM-DD)
          - First Entry Time
          - Last Exit Time
          - Total Time Outside (seconds)
          - Unresolved/Uncertain Crossings (count of needs_review crossings)

        This is the report format an HR/payroll system will consume.

        Args:
            csv_path: Output CSV file path.
        """
        # Aggregate crossing events per person per day
        rows = self.conn.execute("""
            SELECT
                person_name,
                DATE(timestamp) as event_date,
                MIN(CASE WHEN event_type = 'entry' THEN timestamp END) as first_entry,
                MAX(CASE WHEN event_type = 'exit' THEN timestamp END) as last_exit,
                SUM(CASE WHEN needs_review = 1 THEN 1 ELSE 0 END) as review_count
            FROM crossing_events
            GROUP BY person_name, DATE(timestamp)
            ORDER BY event_date ASC, person_name ASC
        """).fetchall()

        # Also compute total time outside per person per day from sessions
        session_rows = self.conn.execute("""
            SELECT
                person_name,
                DATE(exit_time) as event_date,
                SUM(CASE WHEN duration_secs IS NOT NULL THEN duration_secs ELSE 0 END) as total_outside_secs
            FROM sessions
            GROUP BY person_name, DATE(exit_time)
        """).fetchall()

        # Build a lookup for total_outside
        outside_lookup = {}
        for sr in session_rows:
            key = (sr["person_name"], sr["event_date"])
            outside_lookup[key] = sr["total_outside_secs"]

        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Daily Attendance Summary"])
            writer.writerow(["Generated", _to_iso(_now_ist())])
            writer.writerow(["Timezone", "IST (UTC+05:30)"])
            writer.writerow([])
            writer.writerow([
                "Person", "Date", "First Entry Time", "Last Exit Time",
                "Total Time Outside (s)", "Unresolved/Uncertain Crossings"
            ])

            for r in rows:
                person = r["person_name"] or "Unknown"
                date = r["event_date"] or "—"
                first_entry = r["first_entry"] or "—"
                last_exit = r["last_exit"] or "—"
                review_count = r["review_count"] or 0
                total_outside = outside_lookup.get(
                    (r["person_name"], r["event_date"]), 0
                )
                total_outside_str = f"{total_outside:.1f}" if total_outside else "0.0"
                writer.writerow([
                    person, date, first_entry, last_exit,
                    total_outside_str, review_count,
                ])

        print(f"  📄 Daily attendance summary exported: {csv_path}")

    def get_last_crossing_event_id(self, track_id: int,
                                    event_type: str) -> Optional[int]:
        """Get the ID of the most recent crossing event for a track.

        Args:
            track_id: Tracking ID.
            event_type: "entry" or "exit".

        Returns:
            The crossing_events.id, or None if not found.
        """
        row = self.conn.execute(
            "SELECT id FROM crossing_events "
            "WHERE track_id = ? AND event_type = ? "
            "ORDER BY id DESC LIMIT 1",
            (track_id, event_type)
        ).fetchone()
        return row["id"] if row else None

    def log_decision_audit(self, crossing_event_id: Optional[int],
                           track_id: int, frames_considered: int,
                           frames_passed_quality: int,
                           rejection_reason_histogram: dict,
                           final_confidence: float,
                           decision_tier: str,
                           possible_tailgate: bool = False):
        """Write a decision audit record for a crossing event.

        Called once per crossing event at the moment the crossing is
        finalized. Provides full traceability for after-the-fact review.

        Args:
            crossing_event_id: FK to crossing_events.id (may be None).
            track_id: Persistent track ID.
            frames_considered: Total frames where recognition was attempted.
            frames_passed_quality: Frames that passed the quality gate.
            rejection_reason_histogram: Dict like {"TOO_BLURRY": 3, "PASS": 7}.
            final_confidence: Track-level confidence score.
            decision_tier: "high", "uncertain", or "unresolved".
            possible_tailgate: True if tailgating was detected.
        """
        import json
        created = _to_iso(_now_ist())
        histogram_json = json.dumps(rejection_reason_histogram)

        self.conn.execute(
            "INSERT INTO decision_audit "
            "(crossing_event_id, track_id, frames_considered, "
            " frames_passed_quality, rejection_reason_histogram, "
            " final_confidence, decision_tier, possible_tailgate, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (crossing_event_id, track_id, frames_considered,
             frames_passed_quality, histogram_json,
             final_confidence, decision_tier,
             1 if possible_tailgate else 0, created)
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
