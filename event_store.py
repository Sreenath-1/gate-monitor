"""
event_store.py
==============
SQLite WAL-mode event store for vehicle ANPR track events.

Stores:
  - Completed track events (plate text, confidence, timestamps, etc.)
  - Vehicle/plate crop images as files on disk (paths in DB)
  - Query support for dashboard and API

Schema:
  events: id, track_id, plate_text, plate_confidence, plate_valid,
          vehicle_type, state_code, needs_review, first_seen, last_seen,
          duration_s, n_frames, n_plate_reads, n_valid_reads,
          crop_path, plate_crop_path, created_at
  reviews: id, event_id, action, reviewer, notes, created_at

Usage:
    store = EventStore("data/events.db", crop_dir="data/crops")
    store.insert_event(track_event)
    recent = store.get_recent_events(limit=20)
    store.close()
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("event_store")


# Schema version for migrations
SCHEMA_VERSION = 1


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    plate_text TEXT NOT NULL DEFAULT '',
    plate_confidence REAL NOT NULL DEFAULT 0.0,
    plate_valid INTEGER NOT NULL DEFAULT 0,
    vehicle_type TEXT NOT NULL DEFAULT '',
    state_code TEXT NOT NULL DEFAULT '',
    needs_review INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0.0,
    n_frames INTEGER NOT NULL DEFAULT 0,
    n_plate_reads INTEGER NOT NULL DEFAULT 0,
    n_valid_reads INTEGER NOT NULL DEFAULT 0,
    crop_path TEXT DEFAULT NULL,
    plate_crop_path TEXT DEFAULT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_text);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_review ON events(needs_review, reviewed);
CREATE INDEX IF NOT EXISTS idx_events_state ON events(state_code);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    action TEXT NOT NULL,  -- 'approve', 'correct', 'reject'
    corrected_plate TEXT DEFAULT NULL,
    reviewer TEXT DEFAULT NULL,
    notes TEXT DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reviews_event ON reviews(event_id);
"""


class EventStore:
    """SQLite WAL-mode event store for vehicle ANPR events.

    Args:
        db_path: Path to SQLite database file.
        crop_dir: Directory to store crop images.
    """

    def __init__(self, db_path: str = "data/events.db", crop_dir: str = "data/crops"):
        self.db_path = db_path
        self.crop_dir = Path(crop_dir)

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(self.crop_dir / "vehicles", exist_ok=True)
        os.makedirs(self.crop_dir / "plates", exist_ok=True)

        # Open database with WAL mode
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        self._conn.execute("PRAGMA busy_timeout=5000")

        # Create tables
        self._conn.executescript(CREATE_TABLES_SQL)

        # Check/set schema version
        cur = self._conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        if row[0] is None:
            self._conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            self._conn.commit()

        logger.info("EventStore opened: %s (WAL mode)", db_path)

    def insert_event(self, event) -> int:
        """Insert a TrackEvent into the store.

        Args:
            event: TrackEvent from vehicle_tracker.

        Returns:
            The inserted event ID.
        """
        # Save crops to disk
        crop_path = None
        plate_crop_path = None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if event.best_crop is not None:
            crop_filename = f"v_{event.track_id}_{ts}.jpg"
            crop_path = str(self.crop_dir / "vehicles" / crop_filename)
            try:
                cv2.imwrite(crop_path, event.best_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception as e:
                logger.warning("Failed to save vehicle crop: %s", e)
                crop_path = None

        if event.plate_crop is not None:
            plate_filename = f"p_{event.track_id}_{ts}.jpg"
            plate_crop_path = str(self.crop_dir / "plates" / plate_filename)
            try:
                cv2.imwrite(plate_crop_path, event.plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:
                logger.warning("Failed to save plate crop: %s", e)
                plate_crop_path = None

        cur = self._conn.execute("""
            INSERT INTO events (
                track_id, plate_text, plate_confidence, plate_valid,
                vehicle_type, state_code, needs_review,
                first_seen, last_seen, duration_s,
                n_frames, n_plate_reads, n_valid_reads,
                crop_path, plate_crop_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.track_id,
            event.plate_text,
            event.plate_confidence,
            1 if event.plate_valid else 0,
            event.vehicle_type,
            event.state_code,
            1 if event.needs_review else 0,
            event.first_seen,
            event.last_seen,
            event.duration_s,
            event.n_frames,
            event.n_plate_reads,
            event.n_valid_reads,
            crop_path,
            plate_crop_path,
        ))
        self._conn.commit()

        event_id = cur.lastrowid
        logger.debug("Event %d inserted: track=%d plate=%s",
                      event_id, event.track_id, event.plate_text)
        return event_id

    def get_recent_events(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """Get recent events ordered by creation time."""
        cur = self._conn.execute("""
            SELECT * FROM events
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        return [dict(row) for row in cur.fetchall()]

    def get_event(self, event_id: int) -> Optional[dict]:
        """Get a single event by ID."""
        cur = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_events_needing_review(self, limit: int = 50) -> List[dict]:
        """Get events that need human review."""
        cur = self._conn.execute("""
            SELECT * FROM events
            WHERE needs_review = 1 AND reviewed = 0
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]

    def search_plate(self, plate_text: str, limit: int = 20) -> List[dict]:
        """Search events by plate text (partial match)."""
        cur = self._conn.execute("""
            SELECT * FROM events
            WHERE plate_text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{plate_text}%", limit))
        return [dict(row) for row in cur.fetchall()]

    def review_event(
        self,
        event_id: int,
        action: str,
        corrected_plate: str = None,
        reviewer: str = None,
        notes: str = None,
    ):
        """Record a review action on an event.

        Args:
            event_id: Event ID.
            action: 'approve', 'correct', or 'reject'.
            corrected_plate: Corrected plate text (for 'correct' action).
            reviewer: Reviewer name/ID.
            notes: Optional notes.
        """
        self._conn.execute("""
            INSERT INTO reviews (event_id, action, corrected_plate, reviewer, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (event_id, action, corrected_plate, reviewer, notes))

        self._conn.execute("""
            UPDATE events SET reviewed = 1 WHERE id = ?
        """, (event_id,))

        # If corrected, update the plate text
        if action == "correct" and corrected_plate:
            self._conn.execute("""
                UPDATE events SET plate_text = ?, plate_valid = 1, needs_review = 0
                WHERE id = ?
            """, (corrected_plate, event_id))

        self._conn.commit()
        logger.info("Event %d reviewed: %s", event_id, action)

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        stats = {}

        cur = self._conn.execute("SELECT COUNT(*) FROM events")
        stats["total_events"] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COUNT(*) FROM events WHERE plate_valid = 1")
        stats["valid_plates"] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE needs_review = 1 AND reviewed = 0"
        )
        stats["pending_review"] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE reviewed = 1"
        )
        stats["reviewed"] = cur.fetchone()[0]

        cur = self._conn.execute("""
            SELECT vehicle_type, COUNT(*) as cnt
            FROM events WHERE vehicle_type != ''
            GROUP BY vehicle_type ORDER BY cnt DESC
        """)
        stats["vehicle_types"] = {row["vehicle_type"]: row["cnt"] for row in cur.fetchall()}

        cur = self._conn.execute("""
            SELECT state_code, COUNT(*) as cnt
            FROM events WHERE state_code != ''
            GROUP BY state_code ORDER BY cnt DESC LIMIT 10
        """)
        stats["top_states"] = {row["state_code"]: row["cnt"] for row in cur.fetchall()}

        return stats

    def get_disk_usage(self) -> dict:
        """Get disk usage for crops."""
        total_bytes = 0
        n_files = 0
        for dirpath, _, filenames in os.walk(self.crop_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_bytes += os.path.getsize(fp)
                n_files += 1

        return {
            "crop_dir": str(self.crop_dir),
            "n_files": n_files,
            "total_mb": round(total_bytes / (1024 * 1024), 1),
        }

    def purge_oldest_crops(self, max_mb: float = 1000) -> int:
        """Purge oldest crop files if total exceeds max_mb.

        Returns number of files deleted.
        """
        usage = self.get_disk_usage()
        if usage["total_mb"] <= max_mb:
            return 0

        # Collect all crop files with timestamps
        files = []
        for dirpath, _, filenames in os.walk(self.crop_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                files.append((os.path.getmtime(fp), fp, os.path.getsize(fp)))

        # Sort oldest first
        files.sort(key=lambda x: x[0])

        deleted = 0
        freed = 0
        target_free = (usage["total_mb"] - max_mb) * 1024 * 1024

        for mtime, fp, size in files:
            if freed >= target_free:
                break
            try:
                os.remove(fp)
                freed += size
                deleted += 1

                # Also null out crop_path in events
                basename = os.path.basename(fp)
                self._conn.execute("""
                    UPDATE events SET crop_path = NULL
                    WHERE crop_path LIKE ?
                """, (f"%{basename}%",))
                self._conn.execute("""
                    UPDATE events SET plate_crop_path = NULL
                    WHERE plate_crop_path LIKE ?
                """, (f"%{basename}%",))

            except OSError as e:
                logger.warning("Failed to delete %s: %s", fp, e)

        if deleted:
            self._conn.commit()
            logger.info("Purged %d crop files (freed %.1f MB)", deleted, freed / (1024*1024))

        return deleted

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("EventStore closed")

    def __del__(self):
        self.close()
