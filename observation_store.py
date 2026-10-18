"""
observation_store.py
====================
Face observation storage for the person search pipeline.

Stores every usable face observation with:
  - Embedding vector (512-d float32, L2-normalized) in a RAM numpy matrix
  - Metadata in SQLite (WAL mode for concurrent read/write)
  - Thumbnail crops on disk (JPEG)
  - Ring buffer of raw frames per camera (JPEG in RAM, configurable minutes)

Designed for:
  - Fast retro search: matrix multiply probe @ observations.T
  - Live watch: compare new embeddings against active probes in real-time
  - Bounded memory: cap RAM matrix rows; spill to SQLite for older obs
  - Restart recovery: reload recent embeddings from SQLite on boot

Thread-safety:
  - ingest workers APPEND observations (write lock on matrix/DB)
  - search threads READ snapshots of the matrix (no lock needed for numpy)
  - SQLite WAL mode allows concurrent readers + single writer
"""

import os
import sys
import time
import uuid
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IST


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION DATA TYPE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Observation:
    """A single face observation from the ingest pipeline."""
    obs_id: str                      # UUID
    camera_id: str
    ts_utc: float                    # Unix timestamp
    ts_ist: str                      # IST formatted string
    track_id: int                    # BoT-SORT track ID
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame coords
    det_score: float
    face_size_px: int                # min(face_w, face_h)
    yaw: float
    pitch: float
    roll: float
    blur_score: float                # Laplacian variance
    brightness: float                # Mean pixel intensity
    quality_score: float             # Combined 0-1 quality
    mask_state: str                  # "masked", "unmasked", "unknown"
    is_ir_grayscale: bool
    embedding: np.ndarray            # 512-d float32 L2-normalized
    periocular_embedding: Optional[np.ndarray]  # Nullable
    crop_path: str                   # Path to saved thumbnail
    frame_ref: Optional[str] = None  # Ring buffer frame key
    is_usable: bool = True           # False for "person present, face unusable"
    reject_reason: str = ""          # "TOO_SMALL", "TOO_BLURRY", etc.


# ─────────────────────────────────────────────────────────────────────────────
#  RING BUFFER (raw frames per camera, JPEG in RAM)
# ─────────────────────────────────────────────────────────────────────────────
class FrameRingBuffer:
    """Per-camera ring buffer of JPEG-compressed frames in RAM.

    Keeps the last `max_seconds` of frames, capped at `max_frames`.
    Used to provide context around hits (show the frame, +/- 10s clip).
    """

    def __init__(self, camera_id: str, max_seconds: float = 120.0,
                 max_frames: int = 600, jpeg_quality: int = 70):
        self.camera_id = camera_id
        self.max_seconds = max_seconds
        self.max_frames = max_frames
        self.jpeg_quality = jpeg_quality
        self._buffer: List[Tuple[float, bytes]] = []  # (timestamp, jpeg_bytes)
        self._lock = threading.Lock()

    def add_frame(self, frame: np.ndarray, ts: float):
        """Add a frame (BGR) to the ring buffer."""
        _, jpeg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        jpeg_bytes = jpeg.tobytes()

        with self._lock:
            self._buffer.append((ts, jpeg_bytes))
            # Trim by time
            cutoff = ts - self.max_seconds
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.pop(0)
            # Trim by count
            while len(self._buffer) > self.max_frames:
                self._buffer.pop(0)

    def get_context_frames(self, ts: float, window_seconds: float = 10.0
                           ) -> List[Tuple[float, np.ndarray]]:
        """Get frames within +/- window_seconds of a timestamp."""
        results = []
        with self._lock:
            for frame_ts, jpeg_bytes in self._buffer:
                if abs(frame_ts - ts) <= window_seconds:
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        results.append((frame_ts, frame))
        return results

    def get_frame_at(self, ts: float) -> Optional[np.ndarray]:
        """Get the closest frame to a timestamp."""
        with self._lock:
            if not self._buffer:
                return None
            closest = min(self._buffer, key=lambda x: abs(x[0] - ts))
            arr = np.frombuffer(closest[1], dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    @property
    def time_span_seconds(self) -> float:
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1][0] - self._buffer[0][0]

    @property
    def memory_bytes(self) -> int:
        with self._lock:
            return sum(len(b) for _, b in self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
#  OBSERVATION STORE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CROPS_DIR = os.path.join(os.path.dirname(__file__), "observation_crops")
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "observations.db")


class ObservationStore:
    """Face observation storage with RAM matrix + SQLite + thumbnails.

    Thread-safe: ingest workers call add_observation(); search threads
    call get_matrix_snapshot() for vectorized cosine search.

    Usage:
        store = ObservationStore()
        store.add_observation(obs)
        ...
        matrix, obs_ids = store.get_matrix_snapshot()
        scores = matrix @ probe_embedding  # vectorized search
    """

    def __init__(self, db_path: str = None, crops_dir: str = None,
                 max_ram_observations: int = 50000,
                 retention_days: int = 7,
                 embedding_dim: int = 512):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.crops_dir = crops_dir or _DEFAULT_CROPS_DIR
        self.max_ram = max_ram_observations
        self.retention_days = retention_days
        self.embedding_dim = embedding_dim

        # RAM matrix for fast search
        self._embeddings: List[np.ndarray] = []  # Each 512-d
        self._peri_embeddings: List[Optional[np.ndarray]] = []
        self._obs_ids: List[str] = []
        self._obs_meta: List[dict] = []  # Lightweight metadata
        self._lock = threading.Lock()

        # Ring buffers per camera
        self.ring_buffers: Dict[str, FrameRingBuffer] = {}

        # Create dirs
        os.makedirs(self.crops_dir, exist_ok=True)

        # Initialize SQLite
        self._init_db()

        # Load recent observations from DB on startup
        self._reload_from_db()

    def _init_db(self):
        """Create SQLite database with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                obs_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                ts_utc REAL NOT NULL,
                ts_ist TEXT NOT NULL,
                track_id INTEGER,
                bbox_x1 INTEGER, bbox_y1 INTEGER,
                bbox_x2 INTEGER, bbox_y2 INTEGER,
                det_score REAL,
                face_size_px INTEGER,
                yaw REAL, pitch REAL, roll REAL,
                blur_score REAL,
                brightness REAL,
                quality_score REAL,
                mask_state TEXT,
                is_ir_grayscale INTEGER,
                embedding BLOB,
                periocular_embedding BLOB,
                crop_path TEXT,
                frame_ref TEXT,
                is_usable INTEGER DEFAULT 1,
                reject_reason TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_camera_ts
            ON observations(camera_id, ts_utc)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_ts
            ON observations(ts_utc)
        """)

        # Operator feedback table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_feedback (
                feedback_id TEXT PRIMARY KEY,
                obs_id TEXT,
                probe_hash TEXT,
                verdict TEXT,  -- 'confirm', 'reject', 'ambiguous'
                operator TEXT,
                ts_utc REAL,
                notes TEXT
            )
        """)

        # Audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_audit (
                audit_id TEXT PRIMARY KEY,
                probe_hash TEXT,
                search_type TEXT,  -- 'retro', 'watch', 'combined'
                cameras_searched TEXT,  -- JSON list
                time_window TEXT,
                ts_utc REAL,
                operator TEXT,
                result_count INTEGER,
                thresholds_used TEXT,  -- JSON
                model_version TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _reload_from_db(self):
        """Reload recent observations into RAM on startup."""
        cutoff = time.time() - (self.retention_days * 86400)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observations WHERE ts_utc > ? AND is_usable = 1 "
            "ORDER BY ts_utc DESC LIMIT ?",
            (cutoff, self.max_ram)
        ).fetchall()
        conn.close()

        if not rows:
            return

        loaded = 0
        for row in reversed(rows):  # Oldest first
            emb_blob = row['embedding']
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32).copy()
            if emb.shape[0] != self.embedding_dim:
                continue

            peri_blob = row['periocular_embedding']
            peri = None
            if peri_blob is not None:
                peri = np.frombuffer(peri_blob, dtype=np.float32).copy()

            meta = {
                'obs_id': row['obs_id'],
                'camera_id': row['camera_id'],
                'ts_utc': row['ts_utc'],
                'ts_ist': row['ts_ist'],
                'track_id': row['track_id'],
                'det_score': row['det_score'],
                'quality_score': row['quality_score'],
                'mask_state': row['mask_state'],
                'crop_path': row['crop_path'],
                'face_size_px': row['face_size_px'],
                'is_ir_grayscale': bool(row['is_ir_grayscale']),
            }

            self._embeddings.append(emb)
            self._peri_embeddings.append(peri)
            self._obs_ids.append(row['obs_id'])
            self._obs_meta.append(meta)
            loaded += 1

        print(f"  [ObsStore] Reloaded {loaded} observations from DB")

    def add_observation(self, obs: Observation):
        """Add an observation to RAM + SQLite + save crop.

        Thread-safe. Called by ingest workers.
        """
        # Save to SQLite
        self._save_to_db(obs)

        # Add to RAM matrix (only usable observations)
        if obs.is_usable and obs.embedding is not None:
            meta = {
                'obs_id': obs.obs_id,
                'camera_id': obs.camera_id,
                'ts_utc': obs.ts_utc,
                'ts_ist': obs.ts_ist,
                'track_id': obs.track_id,
                'det_score': obs.det_score,
                'quality_score': obs.quality_score,
                'mask_state': obs.mask_state,
                'crop_path': obs.crop_path,
                'face_size_px': obs.face_size_px,
                'is_ir_grayscale': obs.is_ir_grayscale,
            }

            with self._lock:
                self._embeddings.append(obs.embedding.astype(np.float32))
                self._peri_embeddings.append(obs.periocular_embedding)
                self._obs_ids.append(obs.obs_id)
                self._obs_meta.append(meta)

                # Trim RAM if over limit
                if len(self._embeddings) > self.max_ram:
                    excess = len(self._embeddings) - self.max_ram
                    self._embeddings = self._embeddings[excess:]
                    self._peri_embeddings = self._peri_embeddings[excess:]
                    self._obs_ids = self._obs_ids[excess:]
                    self._obs_meta = self._obs_meta[excess:]

    def _save_to_db(self, obs: Observation):
        """Persist observation to SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            emb_blob = obs.embedding.astype(np.float32).tobytes() if obs.embedding is not None else None
            peri_blob = (obs.periocular_embedding.astype(np.float32).tobytes()
                         if obs.periocular_embedding is not None else None)
            conn.execute("""
                INSERT OR REPLACE INTO observations
                (obs_id, camera_id, ts_utc, ts_ist, track_id,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 det_score, face_size_px, yaw, pitch, roll,
                 blur_score, brightness, quality_score,
                 mask_state, is_ir_grayscale,
                 embedding, periocular_embedding,
                 crop_path, frame_ref, is_usable, reject_reason)
                VALUES (?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?, ?,?,?, ?,?, ?,?, ?,?, ?,?)
            """, (
                obs.obs_id, obs.camera_id, obs.ts_utc, obs.ts_ist,
                obs.track_id,
                obs.bbox[0], obs.bbox[1], obs.bbox[2], obs.bbox[3],
                obs.det_score, obs.face_size_px,
                obs.yaw, obs.pitch, obs.roll,
                obs.blur_score, obs.brightness, obs.quality_score,
                obs.mask_state, int(obs.is_ir_grayscale),
                emb_blob, peri_blob,
                obs.crop_path, obs.frame_ref,
                int(obs.is_usable), obs.reject_reason,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_matrix_snapshot(self) -> Tuple[Optional[np.ndarray],
                                           List[str], List[dict]]:
        """Get a snapshot of the embedding matrix for vectorized search.

        Returns:
            (matrix, obs_ids, obs_meta) where matrix is (N, 512) float32.
            Returns (None, [], []) if no observations.

        Thread-safe: copies the list references; numpy is immutable per-row.
        """
        with self._lock:
            if not self._embeddings:
                return None, [], []
            matrix = np.vstack(self._embeddings)
            ids = list(self._obs_ids)
            meta = list(self._obs_meta)
        return matrix, ids, meta

    def get_periocular_matrix(self) -> Tuple[Optional[np.ndarray],
                                              List[str], List[int]]:
        """Get periocular embeddings (only rows that have them).

        Returns:
            (matrix, obs_ids, indices_into_main)
        """
        with self._lock:
            indices = []
            embs = []
            ids = []
            for i, (peri, oid) in enumerate(
                    zip(self._peri_embeddings, self._obs_ids)):
                if peri is not None:
                    embs.append(peri)
                    ids.append(oid)
                    indices.append(i)
            if not embs:
                return None, [], []
            return np.vstack(embs), ids, indices

    def get_observation_meta(self, obs_id: str) -> Optional[dict]:
        """Get metadata for a specific observation."""
        with self._lock:
            for m in self._obs_meta:
                if m['obs_id'] == obs_id:
                    return m.copy()
        return None

    def get_observations_in_window(self, camera_id: str = None,
                                    start_ts: float = None,
                                    end_ts: float = None) -> List[dict]:
        """Query observations within a time window from SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM observations WHERE is_usable = 1"
        params = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if start_ts:
            query += " AND ts_utc >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND ts_utc <= ?"
            params.append(end_ts)

        query += " ORDER BY ts_utc DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_ring_buffer(self, camera_id: str) -> FrameRingBuffer:
        """Get or create ring buffer for a camera."""
        if camera_id not in self.ring_buffers:
            self.ring_buffers[camera_id] = FrameRingBuffer(camera_id)
        return self.ring_buffers[camera_id]

    def cleanup_old_data(self):
        """Remove observations older than retention_days."""
        cutoff = time.time() - (self.retention_days * 86400)

        # Clean SQLite
        conn = sqlite3.connect(self.db_path)
        # Get crop paths to delete
        rows = conn.execute(
            "SELECT crop_path FROM observations WHERE ts_utc < ?",
            (cutoff,)
        ).fetchall()
        conn.execute("DELETE FROM observations WHERE ts_utc < ?", (cutoff,))
        conn.commit()
        conn.close()

        # Delete crop files
        for (path,) in rows:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Trim RAM
        with self._lock:
            keep = []
            for i, m in enumerate(self._obs_meta):
                if m['ts_utc'] >= cutoff:
                    keep.append(i)
            if len(keep) < len(self._obs_meta):
                self._embeddings = [self._embeddings[i] for i in keep]
                self._peri_embeddings = [self._peri_embeddings[i] for i in keep]
                self._obs_ids = [self._obs_ids[i] for i in keep]
                self._obs_meta = [self._obs_meta[i] for i in keep]

    def save_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  camera_id: str, obs_id: str) -> str:
        """Save a face crop thumbnail to disk.

        Returns the saved file path.
        """
        x1, y1, x2, y2 = bbox
        # Add 20% padding
        h, w = y2 - y1, x2 - x1
        pad_y, pad_x = int(h * 0.2), int(w * 0.2)
        cy1 = max(0, y1 - pad_y)
        cy2 = min(frame.shape[0], y2 + pad_y)
        cx1 = max(0, x1 - pad_x)
        cx2 = min(frame.shape[1], x2 + pad_x)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]

        # Save under camera_id subdirectory
        cam_dir = os.path.join(self.crops_dir, camera_id)
        os.makedirs(cam_dir, exist_ok=True)
        path = os.path.join(cam_dir, f"{obs_id}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    @property
    def observation_count(self) -> int:
        """Number of observations in RAM."""
        return len(self._embeddings)

    def stats(self) -> dict:
        """Return store statistics."""
        with self._lock:
            ram_count = len(self._embeddings)
            peri_count = sum(1 for p in self._peri_embeddings if p is not None)

        ring_stats = {}
        for cid, rb in self.ring_buffers.items():
            ring_stats[cid] = {
                'frames': rb.frame_count,
                'span_s': round(rb.time_span_seconds, 1),
                'memory_mb': round(rb.memory_bytes / 1048576, 1),
            }

        return {
            'ram_observations': ram_count,
            'periocular_count': peri_count,
            'ring_buffers': ring_stats,
            'db_path': self.db_path,
            'crops_dir': self.crops_dir,
        }

    def log_search(self, probe_hash: str, search_type: str,
                   cameras: List[str], time_window: str,
                   result_count: int, thresholds: dict,
                   model_version: str, operator: str = "local"):
        """Log a search to the audit table."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO search_audit
            (audit_id, probe_hash, search_type, cameras_searched,
             time_window, ts_utc, operator, result_count,
             thresholds_used, model_version)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()), probe_hash, search_type,
            json.dumps(cameras), time_window,
            time.time(), operator, result_count,
            json.dumps(thresholds), model_version,
        ))
        conn.commit()
        conn.close()

    def log_feedback(self, obs_id: str, probe_hash: str,
                     verdict: str, operator: str = "local",
                     notes: str = ""):
        """Log operator feedback on a search result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO search_feedback
            (feedback_id, obs_id, probe_hash, verdict, operator, ts_utc, notes)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(uuid.uuid4()), obs_id, probe_hash,
            verdict, operator, time.time(), notes,
        ))
        conn.commit()
        conn.close()
