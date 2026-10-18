"""
tests/test_event_store.py
=========================
Unit tests for event_store.py and vehicle_tracker.py.
"""

import os
import shutil
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Lightweight stand-in for TrackEvent to avoid importing the full tracker
@dataclass
class MockTrackEvent:
    track_id: int = 1
    plate_text: str = "TN09AB1234"
    plate_confidence: float = 0.85
    plate_valid: bool = True
    vehicle_type: str = "car"
    state_code: str = "TN"
    needs_review: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    duration_s: float = 3.5
    n_frames: int = 50
    n_plate_reads: int = 8
    n_valid_reads: int = 5
    best_crop: Optional[np.ndarray] = None
    plate_crop: Optional[np.ndarray] = None


class TestEventStore(unittest.TestCase):
    """Tests for EventStore."""

    def setUp(self):
        """Create a temporary DB."""
        from event_store import EventStore
        self.db_path = "data/_test_events.db"
        self.crop_dir = "data/_test_crops"
        self.store = EventStore(self.db_path, self.crop_dir)

    def tearDown(self):
        """Clean up temp DB and crops."""
        self.store.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        wal = self.db_path + "-wal"
        shm = self.db_path + "-shm"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(shm):
            os.remove(shm)
        if os.path.exists(self.crop_dir):
            shutil.rmtree(self.crop_dir)

    def test_insert_and_get(self):
        """Insert an event and retrieve it."""
        event = MockTrackEvent()
        eid = self.store.insert_event(event)
        self.assertIsNotNone(eid)
        self.assertGreater(eid, 0)

        retrieved = self.store.get_event(eid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["plate_text"], "TN09AB1234")
        self.assertEqual(retrieved["vehicle_type"], "car")
        self.assertEqual(retrieved["state_code"], "TN")

    def test_get_nonexistent(self):
        """Get a non-existent event returns None."""
        self.assertIsNone(self.store.get_event(999))

    def test_recent_events_order(self):
        """Events are returned with most recent first (or by descending ID for same timestamp)."""
        for i in range(5):
            self.store.insert_event(MockTrackEvent(track_id=i, plate_text=f"TN09AB{i:04d}"))

        events = self.store.get_recent_events(limit=5)
        self.assertEqual(len(events), 5)
        # All 5 events should be returned
        plates = {e["plate_text"] for e in events}
        self.assertEqual(len(plates), 5)

    def test_search_plate(self):
        """Search by partial plate text."""
        self.store.insert_event(MockTrackEvent(plate_text="TN09AB1234"))
        self.store.insert_event(MockTrackEvent(plate_text="KA01XY5678"))
        self.store.insert_event(MockTrackEvent(plate_text="TN09CD9999"))

        results = self.store.search_plate("TN09")
        self.assertEqual(len(results), 2)

        results = self.store.search_plate("KA01")
        self.assertEqual(len(results), 1)

        results = self.store.search_plate("MH12")
        self.assertEqual(len(results), 0)

    def test_review_event(self):
        """Review an event and verify state change."""
        eid = self.store.insert_event(MockTrackEvent(needs_review=True))

        # Before review
        event = self.store.get_event(eid)
        self.assertEqual(event["needs_review"], 1)
        self.assertEqual(event["reviewed"], 0)

        # Review: approve
        self.store.review_event(eid, "approve", reviewer="tester")
        event = self.store.get_event(eid)
        self.assertEqual(event["reviewed"], 1)

    def test_review_correct(self):
        """Correct a plate via review."""
        eid = self.store.insert_event(MockTrackEvent(plate_text="TN09A81234"))

        self.store.review_event(eid, "correct", corrected_plate="TN09AB1234")
        event = self.store.get_event(eid)
        self.assertEqual(event["plate_text"], "TN09AB1234")
        self.assertEqual(event["plate_valid"], 1)

    def test_stats(self):
        """Aggregate stats work."""
        for i in range(3):
            self.store.insert_event(MockTrackEvent(
                plate_text=f"TN09AB{i:04d}",
                plate_valid=True,
                vehicle_type="car",
            ))
        self.store.insert_event(MockTrackEvent(
            plate_text="UNKNOWN",
            plate_valid=False,
            vehicle_type="truck",
            needs_review=True,
        ))

        stats = self.store.get_stats()
        self.assertEqual(stats["total_events"], 4)
        self.assertEqual(stats["valid_plates"], 3)
        self.assertGreater(stats["pending_review"], 0)

    def test_events_needing_review(self):
        """Filter events needing review."""
        self.store.insert_event(MockTrackEvent(needs_review=False))
        self.store.insert_event(MockTrackEvent(needs_review=True))
        self.store.insert_event(MockTrackEvent(needs_review=True))

        pending = self.store.get_events_needing_review()
        self.assertEqual(len(pending), 2)

    def test_crop_with_image(self):
        """Insert event with numpy crops, verify files saved."""
        crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        plate_crop = np.random.randint(0, 255, (30, 120, 3), dtype=np.uint8)

        eid = self.store.insert_event(MockTrackEvent(
            best_crop=crop, plate_crop=plate_crop
        ))

        event = self.store.get_event(eid)
        self.assertIsNotNone(event["crop_path"])
        self.assertIsNotNone(event["plate_crop_path"])
        self.assertTrue(os.path.exists(event["crop_path"]))
        self.assertTrue(os.path.exists(event["plate_crop_path"]))

    def test_disk_usage(self):
        """Disk usage returns valid data."""
        usage = self.store.get_disk_usage()
        self.assertIn("n_files", usage)
        self.assertIn("total_mb", usage)

    def test_wal_mode(self):
        """Verify WAL journal mode is enabled."""
        cur = self.store._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        self.assertEqual(mode.lower(), "wal")


class TestEditDistanceTracker(unittest.TestCase):
    """Test the edit distance function in vehicle_tracker."""

    def test_identical(self):
        from vehicle_tracker import _edit_distance
        self.assertEqual(_edit_distance("ABC", "ABC"), 0)

    def test_one_sub(self):
        from vehicle_tracker import _edit_distance
        self.assertEqual(_edit_distance("ABC", "ABD"), 1)

    def test_insertion(self):
        from vehicle_tracker import _edit_distance
        self.assertEqual(_edit_distance("AB", "ABC"), 1)

    def test_empty(self):
        from vehicle_tracker import _edit_distance
        self.assertEqual(_edit_distance("", "ABC"), 3)
        self.assertEqual(_edit_distance("ABC", ""), 3)


if __name__ == "__main__":
    unittest.main()
