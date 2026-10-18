"""
dashboard.py
============
FastAPI dashboard + API for vehicle ANPR pipeline.

Endpoints:
  GET  /api/health         System health (GPU, disk, battery, pipeline stats)
  GET  /api/events         Recent events (paginated)
  GET  /api/events/:id     Single event details
  GET  /api/events/review  Events needing review
  GET  /api/search?plate=  Search events by plate text
  POST /api/events/:id/review  Review an event (approve/correct/reject)
  GET  /api/stats          Aggregate statistics
  GET  /api/stream         SSE stream of new events
  GET  /api/crops/:path    Serve crop images
  GET  /                   Dashboard HTML page

Usage:
    From vehicle_pipeline.py or standalone:
        dashboard = Dashboard(event_store, pipeline)
        dashboard.start(host="0.0.0.0", port=8000)
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("dashboard")

app = FastAPI(
    title="Vehicle ANPR Dashboard",
    version="1.0.0",
    description="Gate Monitor — Vehicle ANPR Pipeline Dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Shared state (set by Dashboard.start())
_event_store = None
_pipeline = None
_sse_clients = []  # List of asyncio.Queue for SSE


def _get_gpu_stats() -> dict:
    """Get GPU stats from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,"
             "power.draw,clocks_throttle_reasons.active,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_temp_c": int(parts[0]),
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "power_w": float(parts[3]) if parts[3] != "[N/A]" else None,
            "throttle_reason": parts[4] if len(parts) > 4 else None,
            "gpu_util_pct": int(parts[5]) if len(parts) > 5 else None,
        }
    except Exception:
        return {}


def _get_disk_free() -> dict:
    """Get disk free space."""
    try:
        usage = psutil.disk_usage(os.getcwd())
        return {
            "total_gb": round(usage.total / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "used_pct": round(usage.percent, 1),
        }
    except Exception:
        return {}


def _get_battery() -> dict:
    """Get battery status."""
    try:
        bat = psutil.sensors_battery()
        if bat:
            return {
                "plugged_in": bat.power_plugged,
                "percent": bat.percent,
                "secs_left": bat.secsleft if bat.secsleft > 0 else None,
            }
    except Exception:
        pass
    return {"plugged_in": None}


# --- API Endpoints ---

@app.get("/api/health")
async def health():
    """System health check."""
    gpu = _get_gpu_stats()
    disk = _get_disk_free()
    battery = _get_battery()

    pipeline_stats = {}
    if _pipeline:
        pipeline_stats = _pipeline.stats

    ram = psutil.virtual_memory()

    return {
        "status": "ok",
        "timestamp": time.time(),
        "gpu": gpu,
        "disk": disk,
        "disk_low": disk.get("free_gb", 999) < 20,
        "battery": battery,
        "ram": {
            "total_mb": round(ram.total / (1024**2)),
            "available_mb": round(ram.available / (1024**2)),
            "used_pct": round(ram.percent, 1),
        },
        "pipeline": pipeline_stats,
    }


@app.get("/api/events")
async def get_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get recent events."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    return _event_store.get_recent_events(limit=limit, offset=offset)


@app.get("/api/events/review")
async def get_events_review(limit: int = Query(50, ge=1, le=200)):
    """Get events needing review."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    return _event_store.get_events_needing_review(limit=limit)


@app.get("/api/events/{event_id}")
async def get_event(event_id: int):
    """Get single event."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    event = _event_store.get_event(event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    return event


class ReviewRequest(BaseModel):
    action: str  # 'approve', 'correct', 'reject'
    corrected_plate: Optional[str] = None
    reviewer: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/events/{event_id}/review")
async def review_event(event_id: int, req: ReviewRequest):
    """Review an event."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    if req.action not in ("approve", "correct", "reject"):
        raise HTTPException(400, "action must be approve, correct, or reject")
    _event_store.review_event(
        event_id, req.action,
        corrected_plate=req.corrected_plate,
        reviewer=req.reviewer,
        notes=req.notes,
    )
    return {"ok": True}


@app.get("/api/search")
async def search(plate: str = Query(..., min_length=1)):
    """Search events by plate text."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    return _event_store.search_plate(plate)


@app.get("/api/stats")
async def get_stats():
    """Aggregate statistics."""
    if not _event_store:
        raise HTTPException(503, "Event store not initialized")
    stats = _event_store.get_stats()
    stats["disk_usage"] = _event_store.get_disk_usage()
    return stats


@app.get("/api/stream")
async def event_stream():
    """SSE stream of new events."""
    queue = asyncio.Queue()
    _sse_clients.append(queue)

    async def generate():
        try:
            # Send heartbeat immediately
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            if queue in _sse_clients:
                _sse_clients.remove(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/crops/{path:path}")
async def serve_crop(path: str):
    """Serve crop images."""
    full_path = Path("data/crops") / path
    if not full_path.exists():
        raise HTTPException(404, "Crop not found")
    return FileResponse(full_path, media_type="image/jpeg")


def broadcast_event(event_data: dict):
    """Broadcast an event to all SSE clients."""
    event_data["type"] = "new_event"
    for queue in _sse_clients:
        try:
            queue.put_nowait(event_data)
        except asyncio.QueueFull:
            pass


# --- Dashboard HTML ---

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gate Monitor — Vehicle ANPR</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f1117;
            --bg-secondary: #1a1d27;
            --bg-card: #222635;
            --bg-card-hover: #2a2f3f;
            --text-primary: #e8eaf0;
            --text-secondary: #9ca3b4;
            --accent: #6c5ce7;
            --accent-glow: rgba(108, 92, 231, 0.3);
            --success: #00d2a0;
            --warning: #f0c040;
            --danger: #ff5c5c;
            --border: rgba(255,255,255,0.06);
            --glass: rgba(30,34,47,0.85);
            --radius: 12px;
            --radius-sm: 8px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .header {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent), #a29bfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .status {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-card);
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .main {
            padding: 24px 32px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .grid-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
        }
        .stat-card .label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        .stat-card .value {
            font-size: 28px;
            font-weight: 700;
        }
        .stat-card .sub {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .events-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 4px;
        }
        .events-table th {
            text-align: left;
            padding: 12px 16px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .events-table td {
            padding: 14px 16px;
            background: var(--bg-card);
            font-size: 14px;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
        }
        .events-table tr td:first-child { border-left: 1px solid var(--border); border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
        .events-table tr td:last-child { border-right: 1px solid var(--border); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }
        .events-table tbody tr { transition: background 0.15s; }
        .events-table tbody tr:hover td { background: var(--bg-card-hover); }
        .plate-text {
            font-family: 'Courier New', monospace;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 1px;
            padding: 4px 10px;
            background: rgba(108,92,231,0.15);
            border: 1px solid rgba(108,92,231,0.3);
            border-radius: 6px;
        }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-valid { background: rgba(0,210,160,0.15); color: var(--success); }
        .badge-review { background: rgba(240,192,64,0.15); color: var(--warning); }
        .badge-invalid { background: rgba(255,92,92,0.15); color: var(--danger); }
        .search-bar {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }
        .search-bar input {
            flex: 1;
            padding: 10px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .search-bar input:focus { border-color: var(--accent); }
        .search-bar button {
            padding: 10px 24px;
            background: var(--accent);
            border: none;
            border-radius: var(--radius-sm);
            color: white;
            font-family: inherit;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .search-bar button:hover { opacity: 0.85; }
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 14px 24px;
            background: var(--bg-card);
            border: 1px solid var(--accent);
            border-radius: var(--radius);
            box-shadow: 0 8px 32px var(--accent-glow);
            font-size: 14px;
            z-index: 200;
            animation: slideIn 0.3s ease-out;
            display: none;
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .temp-warn { color: var(--danger); }
        .temp-ok { color: var(--success); }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 Gate Monitor — Vehicle ANPR</h1>
        <div class="status">
            <div class="status-badge">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
            <div class="status-badge" id="fpsDisplay">-- FPS</div>
            <div class="status-badge" id="gpuDisplay">GPU: --°C</div>
        </div>
    </div>

    <div class="main">
        <div class="grid-stats" id="statsGrid">
            <div class="stat-card"><div class="label">Total Events</div><div class="value" id="statTotal">--</div></div>
            <div class="stat-card"><div class="label">Valid Plates</div><div class="value" id="statValid">--</div></div>
            <div class="stat-card"><div class="label">Pending Review</div><div class="value" id="statReview">--</div><div class="sub" style="color:var(--warning)">needs attention</div></div>
            <div class="stat-card"><div class="label">Active Tracks</div><div class="value" id="statTracks">--</div></div>
            <div class="stat-card"><div class="label">VRAM</div><div class="value" id="statVram">--</div><div class="sub">MB used</div></div>
            <div class="stat-card"><div class="label">Disk Free</div><div class="value" id="statDisk">--</div><div class="sub">GB</div></div>
        </div>

        <div class="search-bar">
            <input type="text" id="searchInput" placeholder="Search plate number (e.g. TN09AB1234)..." />
            <button onclick="searchPlate()">Search</button>
        </div>

        <div class="section-title">📋 Recent Events</div>
        <table class="events-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Time</th>
                    <th>Plate</th>
                    <th>Confidence</th>
                    <th>Vehicle</th>
                    <th>State</th>
                    <th>Duration</th>
                    <th>Reads</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="eventsBody"></tbody>
        </table>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let eventSource = null;

        function formatTime(ts) {
            if (!ts) return '--';
            const d = new Date(ts);
            if (isNaN(d.getTime())) return ts.substring(0, 19);
            return d.toLocaleTimeString();
        }

        function renderEvent(ev) {
            const conf = (ev.plate_confidence * 100).toFixed(0);
            let statusBadge;
            if (ev.plate_valid && !ev.needs_review) {
                statusBadge = '<span class="badge badge-valid">Valid</span>';
            } else if (ev.needs_review && !ev.reviewed) {
                statusBadge = '<span class="badge badge-review">Review</span>';
            } else if (ev.reviewed) {
                statusBadge = '<span class="badge badge-valid">Reviewed</span>';
            } else {
                statusBadge = '<span class="badge badge-invalid">Invalid</span>';
            }
            return `<tr>
                <td>${ev.id}</td>
                <td>${formatTime(ev.created_at)}</td>
                <td><span class="plate-text">${ev.plate_text || '—'}</span></td>
                <td>${conf}%</td>
                <td>${ev.vehicle_type || '—'}</td>
                <td>${ev.state_code || '—'}</td>
                <td>${(ev.duration_s || 0).toFixed(1)}s</td>
                <td>${ev.n_plate_reads || 0}</td>
                <td>${statusBadge}</td>
            </tr>`;
        }

        async function loadEvents() {
            try {
                const res = await fetch('/api/events?limit=30');
                const events = await res.json();
                document.getElementById('eventsBody').innerHTML =
                    events.map(renderEvent).join('');
            } catch (e) {
                console.error('Load events failed:', e);
            }
        }

        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const s = await res.json();
                document.getElementById('statTotal').textContent = s.total_events || 0;
                document.getElementById('statValid').textContent = s.valid_plates || 0;
                document.getElementById('statReview').textContent = s.pending_review || 0;
            } catch (e) {}
        }

        async function loadHealth() {
            try {
                const res = await fetch('/api/health');
                const h = await res.json();
                const gpu = h.gpu || {};
                const el = document.getElementById('gpuDisplay');
                if (gpu.gpu_temp_c) {
                    el.textContent = `GPU: ${gpu.gpu_temp_c}°C`;
                    el.className = 'status-badge ' + (gpu.gpu_temp_c > 80 ? 'temp-warn' : 'temp-ok');
                }
                if (h.pipeline && h.pipeline.fps) {
                    document.getElementById('fpsDisplay').textContent = h.pipeline.fps.toFixed(1) + ' FPS';
                    document.getElementById('statTracks').textContent = h.pipeline.active_tracks || 0;
                    document.getElementById('statVram').textContent = h.pipeline.vram_mb || '--';
                }
                if (h.disk) {
                    document.getElementById('statDisk').textContent = h.disk.free_gb || '--';
                }
            } catch (e) {}
        }

        async function searchPlate() {
            const q = document.getElementById('searchInput').value.trim();
            if (!q) return;
            try {
                const res = await fetch('/api/search?plate=' + encodeURIComponent(q));
                const events = await res.json();
                document.getElementById('eventsBody').innerHTML =
                    events.length > 0
                        ? events.map(renderEvent).join('')
                        : '<tr><td colspan="9" style="text-align:center;color:var(--text-secondary)">No results</td></tr>';
            } catch (e) {
                console.error('Search failed:', e);
            }
        }

        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchPlate();
        });

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.display = 'block';
            setTimeout(() => { t.style.display = 'none'; }, 4000);
        }

        function connectSSE() {
            eventSource = new EventSource('/api/stream');
            eventSource.onopen = () => {
                document.getElementById('statusDot').style.background = 'var(--success)';
                document.getElementById('statusText').textContent = 'Connected';
            };
            eventSource.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.type === 'new_event') {
                        loadEvents();
                        loadStats();
                        showToast(`New vehicle: ${data.plate_text || 'Unknown plate'} (${data.vehicle_type || ''})`);
                    }
                } catch (err) {}
            };
            eventSource.onerror = () => {
                document.getElementById('statusDot').style.background = 'var(--danger)';
                document.getElementById('statusText').textContent = 'Reconnecting...';
                setTimeout(connectSSE, 3000);
            };
        }

        // Initial load
        loadEvents();
        loadStats();
        loadHealth();
        connectSSE();

        // Periodic refresh
        setInterval(loadHealth, 5000);
        setInterval(loadStats, 10000);
        setInterval(loadEvents, 15000);
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


class Dashboard:
    """Dashboard server wrapper.

    Args:
        event_store: EventStore instance.
        pipeline: VehiclePipeline instance (for stats).
        host: Bind host.
        port: Bind port.
    """

    def __init__(self, event_store, pipeline=None, host="0.0.0.0", port=8000):
        global _event_store, _pipeline
        _event_store = event_store
        _pipeline = pipeline
        self.host = host
        self.port = port

    def start(self, block=False):
        """Start the dashboard server.

        Args:
            block: If True, blocks forever. If False, runs in a background thread.
        """
        import uvicorn

        if block:
            uvicorn.run(app, host=self.host, port=self.port, log_level="info")
        else:
            import threading
            t = threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={"host": self.host, "port": self.port, "log_level": "warning"},
                daemon=True,
            )
            t.start()
            logger.info("Dashboard started on http://%s:%d", self.host, self.port)

    def set_pipeline(self, pipeline):
        """Update the pipeline reference (for late binding)."""
        global _pipeline
        _pipeline = pipeline
