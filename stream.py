"""
stream.py
=========
Video source handling — RTSP streams, local video files, and webcams.

Key features:
  - ThreadedCapture: background thread keeps RTSP alive while inference runs
  - Auto TCP transport for RTSP (avoids UDP packet loss)
  - Connection health checking and frame reading
  - Resize support for high-resolution streams (3MP → 720p for CPU speed)
  - RTSP reconnection: ThreadedCapture automatically reconnects on connection
    drop (tested by simulating disconnect). Tracks consecutive failures and
    reconnects with exponential backoff up to a configurable max.

The ThreadedCapture solves a critical RTSP problem:
  YOLO inference takes ~0.3-1.2s/frame on CPU, but RTSP streams expect
  continuous frame reads or the connection times out. A daemon thread
  continuously reads frames (keeping the connection alive) and stores
  only the latest one.
"""

import os
import time
import threading

import cv2

from config import RTSP_RECONNECT_DELAY, RTSP_MAX_RECONNECT_SECS


def is_stream_source(source_str):
    """Detect if source is a live stream (RTSP/HTTP) vs a local file or webcam index.

    Returns True for RTSP/HTTP URLs, False for file paths and webcam indices.
    """
    s = str(source_str).lower()
    return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://")


def open_capture(source, is_stream):
    """Open a cv2.VideoCapture with stream-optimized settings.

    For RTSP streams:
      - Uses FFMPEG backend (more reliable than default GStreamer/DirectShow)
      - Forces TCP transport to avoid UDP packet loss
      - Sets buffer size to 1 to always get the latest frame
    """
    if is_stream:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    return cap


class ThreadedCapture:
    """Background-threaded frame reader for live streams with auto-reconnect.

    A daemon thread continuously reads frames (keeping the RTSP connection
    alive) and stores only the latest one. The main loop calls read() to
    grab the most recent frame whenever it's ready to process. Old frames
    are silently discarded — we always want the latest view.

    Reconnection logic:
      If cap.read() fails repeatedly, the thread attempts to reopen the
      connection with exponential backoff (starting at RTSP_RECONNECT_DELAY
      seconds, doubling each attempt). After RTSP_MAX_RECONNECT_SECS total
      seconds of failures, it gives up and sets self.connection_lost = True.

      This allows the system to survive:
        - Brief network glitches (router hiccup, cable reseat)
        - Camera firmware restarts
        - Router reboots (JioFiber, in this case)

    Usage:
        cap = ThreadedCapture(source, is_stream=True)
        while True:
            ret, frame = cap.read()
            if cap.connection_lost:
                print("Camera disconnected permanently")
                break
            if not ret:
                time.sleep(0.05)
                continue
            # process frame...
        cap.release()
    """

    def __init__(self, source, is_stream):
        self.source = source
        self.is_stream = is_stream
        self.cap = open_capture(source, is_stream)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.connection_lost = False

        # Reconnection tracking
        self._consecutive_failures = 0
        self._reconnect_start_time = None
        self._total_reconnects = 0

        # Read initial properties before starting background thread
        self._initial_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 0
        self._initial_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 0
        self._initial_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.isOpened() else 0

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reconnect(self):
        """Attempt to reconnect to the stream with exponential backoff.

        Returns True if reconnection succeeded, False if max time exceeded.
        """
        if self._reconnect_start_time is None:
            self._reconnect_start_time = time.time()

        elapsed = time.time() - self._reconnect_start_time
        if elapsed > RTSP_MAX_RECONNECT_SECS:
            print(f"  ✗ RTSP reconnect failed after {elapsed:.0f}s "
                  f"({self._total_reconnects} attempts). Giving up.")
            return False

        # Exponential backoff: 2s, 4s, 8s... capped at 16s
        delay = min(RTSP_RECONNECT_DELAY * (2 ** min(self._total_reconnects, 3)), 16)
        self._total_reconnects += 1
        print(f"  ⟳ RTSP reconnecting (attempt {self._total_reconnects}, "
              f"backoff {delay:.0f}s, elapsed {elapsed:.0f}s)...")

        time.sleep(delay)

        try:
            self.cap.release()
        except Exception:
            pass

        new_cap = open_capture(self.source, self.is_stream)
        if new_cap.isOpened():
            ret, frame = new_cap.read()
            if ret:
                with self.lock:
                    self.cap = new_cap
                    self.ret = ret
                    self.frame = frame
                self._consecutive_failures = 0
                self._reconnect_start_time = None
                print(f"  ✓ RTSP reconnected successfully after "
                      f"{self._total_reconnects} attempt(s)")
                return True
            else:
                new_cap.release()
        return False  # Will retry on next iteration

    def _reader(self):
        """Continuously read frames in background, keeping only the latest.

        Auto-reconnects on connection drop with exponential backoff.
        Catches cv2.error on shutdown so the thread exits cleanly when
        the capture is released while we're mid-read.
        """
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                # Capture was released (shutdown) — exit cleanly
                if self.stopped:
                    return
                ret, frame = False, None
            except Exception:
                if self.stopped:
                    return
                ret, frame = False, None

            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                self._consecutive_failures = 0
                self._reconnect_start_time = None
            else:
                if self.stopped:
                    return  # Don't reconnect during shutdown
                self._consecutive_failures += 1

                if self._consecutive_failures >= 10:
                    # Connection appears dropped — attempt reconnect
                    if not self._reconnect():
                        if (self._reconnect_start_time and
                            time.time() - self._reconnect_start_time > RTSP_MAX_RECONNECT_SECS):
                            self.connection_lost = True
                            with self.lock:
                                self.ret = False
                            return  # Stop the reader thread
                else:
                    time.sleep(0.01)

    def read(self):
        """Return the latest frame (thread-safe). Returns (bool, frame_or_None)."""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        """Get capture property. Uses cached values for resolution/FPS to avoid
        issues when the capture is being reconnected."""
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._initial_width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._initial_height
        elif prop == cv2.CAP_PROP_FPS:
            return self._initial_fps
        return self.cap.get(prop)

    @property
    def reconnect_count(self):
        """Total number of reconnection attempts made."""
        return self._total_reconnects

    def release(self):
        """Stop the reader thread and release the capture.

        Ordering matters: set stopped flag first so the reader thread
        exits its loop, then join the thread (with timeout) so it's
        not mid-read when we release the capture.
        """
        self.stopped = True
        # Wait for reader thread to finish (up to 2s)
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


class FrameSource:
    """Unified frame source that handles both streams and video files.

    Encapsulates:
      - Stream vs file detection
      - ThreadedCapture for streams, plain cv2.VideoCapture for files
      - Frame resizing for high-resolution sources
      - Resolution and FPS reporting

    Usage:
        src = FrameSource("rtsp://...", resize_height=720)
        while True:
            ret, frame = src.read()
            if not ret:
                if src.connection_lost:
                    break
                time.sleep(0.05)
                continue
            # process frame...
        src.release()
    """

    def __init__(self, source, resize_height=720):
        """
        Args:
            source: Video file path, RTSP URL, or webcam index string ("0").
            resize_height: Target frame height for processing. 0 = no resize.
        """
        self.source_str = source
        self.is_stream = is_stream_source(source)

        # Parse webcam index
        parsed = int(source) if source.isdigit() else source

        # Open capture
        if self.is_stream:
            self.cap = ThreadedCapture(parsed, self.is_stream)
        else:
            self.cap = open_capture(parsed, self.is_stream)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Read native properties
        self.raw_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.raw_fps    = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sanitize FPS for streams (some cameras report 0 or garbage values)
        if self.is_stream and (self.raw_fps <= 0 or self.raw_fps > 120):
            self.raw_fps = 25.0

        # Resize configuration
        self.do_resize = resize_height > 0 and self.raw_height > resize_height
        if self.do_resize:
            scale = resize_height / self.raw_height
            self.frame_width  = int(self.raw_width * scale)
            self.frame_height = resize_height
        else:
            self.frame_width  = self.raw_width
            self.frame_height = self.raw_height

    @property
    def connection_lost(self) -> bool:
        """Check if the RTSP connection has been permanently lost."""
        if hasattr(self.cap, 'connection_lost'):
            return self.cap.connection_lost
        return False

    def read(self):
        """Read and optionally resize one frame.

        Returns:
            (bool, frame_or_None): Success flag and the (possibly resized) frame.
        """
        ret, frame = self.cap.read()
        if ret and frame is not None and self.do_resize:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        return ret, frame

    def info_dict(self):
        """Return a dict of source properties for logging/display."""
        return {
            "source":       self.source_str,
            "mode":         "LIVE STREAM" if self.is_stream else "Video File",
            "raw_resolution": f"{self.raw_width}×{self.raw_height}",
            "process_resolution": f"{self.frame_width}×{self.frame_height}",
            "resized":      self.do_resize,
            "fps":          self.raw_fps,
            "total_frames": self.total_frames if not self.is_stream else "∞",
        }

    def release(self):
        """Release the underlying capture."""
        self.cap.release()


def check_connection(source, num_frames=10, timeout=15):
    """Quick health check: open source, read N frames, report timing.

    Used during Stage 0 plumbing check to confirm RTSP/video works.

    Args:
        source: Video file path or RTSP URL.
        num_frames: Number of frames to read.
        timeout: Max seconds before giving up.

    Returns:
        dict with 'success', 'frames_read', 'elapsed', 'fps', 'resolution', 'error'.
    """
    result = {
        "success": False,
        "frames_read": 0,
        "elapsed": 0.0,
        "fps": 0.0,
        "resolution": "",
        "error": None,
    }

    try:
        is_stream = is_stream_source(source)
        cap = open_capture(source if not str(source).isdigit() else int(source), is_stream)

        if not cap.isOpened():
            result["error"] = f"Cannot open: {source}"
            return result

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["resolution"] = f"{w}×{h}"

        t_start = time.time()
        for i in range(num_frames):
            if time.time() - t_start > timeout:
                result["error"] = f"Timeout after {timeout}s (read {i} frames)"
                break
            ret, frame = cap.read()
            if not ret:
                result["error"] = f"Read failed at frame {i}"
                break
            result["frames_read"] += 1

        result["elapsed"] = time.time() - t_start
        if result["elapsed"] > 0 and result["frames_read"] > 0:
            result["fps"] = result["frames_read"] / result["elapsed"]
        result["success"] = result["frames_read"] >= num_frames

        cap.release()

    except Exception as e:
        result["error"] = str(e)

    return result
