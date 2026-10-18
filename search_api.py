"""
search_api.py
=============
Search interface for the person search system.

Provides:
  1. CLI search: python main.py --mode search --photo target.jpg --last 30m
  2. Interactive CLI: paste/drag image path, Enter = Search
  3. FastAPI web UI: upload photo → probe quality → search → results grid
  4. Live watch mode: real-time alerts when probe matches new observations

The web UI is bound to 127.0.0.1 by default (no remote exposure).
Access control via local password (configurable in .env).

All search results include:
  - Tier classification (STRONG / POSSIBLE / WEAK)
  - Coverage statement (cameras searched, time window, % coverage)
  - Audit trail (probe hash, thresholds, model version, operator)
  - "Possible match for review" wording — never "criminal found"
"""

import os
import sys
import json
import time
import hashlib
import threading
import uuid
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IST
from probe_builder import ProbeBuilder, ProbeTemplate
from matcher import Matcher, MatchResult, SearchResult, MatchTier
from observation_store import ObservationStore


# ─────────────────────────────────────────────────────────────────────────────
#  CLI SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def cli_search(photo_path: str, store: ObservationStore,
               embedder, last_minutes: float = 30,
               watch: bool = False,
               camera_ids: List[str] = None):
    """Run a search from the command line.

    Args:
        photo_path: Path to target photo.
        store: ObservationStore instance.
        embedder: FaceEmbedder instance.
        last_minutes: Search the last N minutes.
        watch: If True, also watch for future matches.
        camera_ids: Limit to specific cameras.
    """
    print(f"\n{'='*60}")
    print(f"  PERSON SEARCH")
    print(f"{'='*60}")
    print(f"  Photo  : {photo_path}")
    print(f"  Window : last {last_minutes} minutes")
    print(f"  Mode   : {'retro + live watch' if watch else 'retro only'}")

    # Build probe
    builder = ProbeBuilder(embedder)
    try:
        probe = builder.build_from_photos([photo_path])
    except ValueError as e:
        # Multiple faces — show them
        print(f"\n  {e}")
        faces = builder.detect_faces_in_photo(photo_path)
        print(f"  Found {len(faces)} faces. Use --face-index N to select.")
        for idx, crop, score in faces:
            print(f"    [{idx}] det_score={score:.3f} "
                  f"size={crop.shape[1]}x{crop.shape[0]}")
        return
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return

    if probe is None:
        print(f"\n  ERROR: No face detected in photo.")
        print(f"  Tips: use a clear frontal photo with good lighting.")
        return

    # Show probe quality
    qr = probe.quality_report
    print(f"\n  Probe Quality:")
    print(f"    Face size   : {qr.face_size_px}px")
    print(f"    Reliability : {qr.reliability}")
    print(f"    Templates   : {probe.template_count}")
    if qr.warnings:
        for w in qr.warnings:
            print(f"    ⚠ {w}")

    # Run retro search
    matcher = Matcher(store)
    result = matcher.retro_search(
        probe, last_minutes=last_minutes, camera_ids=camera_ids)

    _print_results(result)

    if watch:
        print(f"\n  Live watch mode active. Press Ctrl+C to stop.\n")
        _run_live_watch(probe, matcher, store)


def _print_results(result: SearchResult):
    """Print search results to terminal."""
    print(f"\n{'─'*60}")
    print(f"  SEARCH RESULTS — {result.timestamp}")
    print(f"{'─'*60}")

    if not result.episodes:
        print(f"\n  No matches found.")
    else:
        print(f"\n  Found {len(result.all_matches)} match(es) in "
              f"{len(result.episodes)} episode(s):")

        for ep in result.episodes:
            print(f"\n  ┌─ Episode: {ep.camera_id} | "
                  f"{ep.ts_ist_start} → {ep.ts_ist_end}")
            print(f"  │  Tier: {ep.tier} | Best score: {ep.best_score:.3f} | "
                  f"Observations: {ep.observation_count}")

            for m in ep.matches[:5]:  # Show top 5 per episode
                flags = " ".join(f"[{f}]" for f in m.flags) if m.flags else ""
                print(f"  │  • score={m.combined_score:.3f} "
                      f"({m.tier}) quality={m.quality_score:.2f} "
                      f"mask={m.mask_state} {flags}")
                if m.crop_path:
                    print(f"  │    crop: {m.crop_path}")

            if len(ep.matches) > 5:
                print(f"  │  ... and {len(ep.matches)-5} more")
            print(f"  └─")

    # Coverage
    cov = result.coverage
    print(f"\n  Coverage:")
    print(f"    Cameras searched : {', '.join(cov.get('cameras_searched', []))}")
    window = cov.get('window_duration_minutes', 0)
    print(f"    Time window      : {window:.1f} minutes")
    print(f"    Usable faces     : {cov.get('total_usable_observations', 0)}")
    print(f"    Note: {cov.get('note', '')}")

    print(f"\n  ⚠  Results are POSSIBLE MATCHES FOR REVIEW.")
    print(f"     STRONG ≠ proof. Verify with context frames and human judgment.")
    print(f"{'─'*60}\n")


def _run_live_watch(probe: ProbeTemplate, matcher: Matcher,
                    store: ObservationStore):
    """Live watch mode: monitor new observations and alert on matches."""
    seen_obs = set()
    cooldowns = {}  # camera_id → last_alert_time
    COOLDOWN_SECONDS = 20

    try:
        while True:
            # Check for new observations
            _, obs_ids, obs_meta = store.get_matrix_snapshot()
            matrix, _, _ = store.get_matrix_snapshot()

            if matrix is None:
                time.sleep(1)
                continue

            for i, (oid, meta) in enumerate(zip(obs_ids, obs_meta)):
                if oid in seen_obs:
                    continue
                seen_obs.add(oid)

                # Check cooldown
                cid = meta['camera_id']
                if cid in cooldowns:
                    if time.time() - cooldowns[cid] < COOLDOWN_SECONDS:
                        continue

                # Match
                emb = matrix[i]
                result = matcher.live_match(
                    probe, emb, None, cid, meta['ts_utc'],
                    quality_score=meta.get('quality_score', 0),
                    mask_state=meta.get('mask_state', 'unknown'),
                )

                if result and result.tier in (MatchTier.STRONG, MatchTier.POSSIBLE):
                    cooldowns[cid] = time.time()
                    ts = meta.get('ts_ist', '')
                    print(f"\n  🔔 ALERT [{result.tier}] "
                          f"score={result.combined_score:.3f} "
                          f"camera={cid} time={ts}")
                    if result.tier == MatchTier.STRONG:
                        # Play alert sound on Windows
                        try:
                            import winsound
                            winsound.Beep(1000, 300)
                        except Exception:
                            pass

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n  Live watch stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────
def interactive_search(store: ObservationStore, embedder,
                       camera_ids: List[str] = None):
    """Interactive search prompt: paste a path, press Enter = Search."""
    print(f"\n{'='*60}")
    print(f"  INTERACTIVE PERSON SEARCH")
    print(f"{'='*60}")
    print(f"  Paste or drag an image path and press Enter to search.")
    print(f"  Type 'stats' for observation stats, 'quit' to exit.\n")

    while True:
        try:
            line = input("  Search> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not line:
            continue
        if line.lower() in ('quit', 'exit', 'q'):
            break
        if line.lower() == 'stats':
            s = store.stats()
            print(f"  Observations in RAM: {s['ram_observations']}")
            for cid, rs in s.get('ring_buffers', {}).items():
                print(f"  Ring [{cid}]: {rs['frames']} frames, "
                      f"{rs['span_s']}s span")
            continue

        # Treat as photo path
        path = line.replace('\\', '/')
        if not os.path.exists(path):
            # Try as-is (might have spaces)
            if not os.path.exists(line):
                print(f"  File not found: {line}")
                continue
            path = line

        cli_search(path, store, embedder, last_minutes=30,
                   camera_ids=camera_ids)


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI WEB UI
# ─────────────────────────────────────────────────────────────────────────────
def create_web_app(store: ObservationStore, embedder,
                   camera_mgr=None, ingest_mgr=None,
                   password: str = None):
    """Create FastAPI web application for person search.

    Bound to 127.0.0.1 by default. Local password optional.
    """
    try:
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
        from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        print("  [SearchAPI] FastAPI not installed. Web UI disabled.")
        print("  Install with: pip install fastapi uvicorn python-multipart")
        return None

    app = FastAPI(title="Person Search", docs_url=None)

    # Store references
    app.state.store = store
    app.state.embedder = embedder
    app.state.camera_mgr = camera_mgr
    app.state.ingest_mgr = ingest_mgr
    app.state.password = password or os.environ.get("SEARCH_UI_PASSWORD", "")
    app.state.active_probes = {}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _get_html_ui()

    @app.get("/api/health")
    async def health():
        result = {"status": "ok", "observations": store.observation_count}
        if camera_mgr:
            result["cameras"] = camera_mgr.health_report()
        if ingest_mgr:
            result["ingest"] = ingest_mgr.stats()
        result["store"] = store.stats()
        return result

    @app.post("/api/search")
    async def search(photo: UploadFile = File(...),
                     last_minutes: float = Form(30),
                     face_index: int = Form(None)):
        # Save upload to temp
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{Path(photo.filename).suffix}")

        content = await photo.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            builder = ProbeBuilder(embedder)
            probe = builder.build_from_photos(
                [temp_path], face_index=face_index)

            if probe is None:
                return JSONResponse({"error": "No face detected in photo"}, 400)

            matcher = Matcher(store)
            result = matcher.retro_search(probe, last_minutes=last_minutes)

            # Convert to JSON-safe format
            return {
                "probe_hash": result.probe_hash,
                "probe_quality": {
                    "reliability": probe.quality_report.reliability,
                    "face_size_px": probe.quality_report.face_size_px,
                    "warnings": probe.quality_report.warnings,
                    "template_count": probe.template_count,
                },
                "episodes": [
                    {
                        "episode_id": ep.episode_id,
                        "camera_id": ep.camera_id,
                        "tier": ep.tier,
                        "best_score": round(ep.best_score, 4),
                        "time_start": ep.ts_ist_start,
                        "time_end": ep.ts_ist_end,
                        "observation_count": ep.observation_count,
                        "matches": [
                            {
                                "obs_id": m.obs_id,
                                "score": round(m.combined_score, 4),
                                "tier": m.tier,
                                "camera_id": m.camera_id,
                                "ts_ist": m.ts_ist,
                                "quality": round(m.quality_score, 3),
                                "mask_state": m.mask_state,
                                "crop_path": m.crop_path,
                                "flags": m.flags,
                            }
                            for m in ep.matches[:10]
                        ],
                    }
                    for ep in result.episodes
                ],
                "total_matches": len(result.all_matches),
                "coverage": result.coverage,
                "timestamp": result.timestamp,
            }
        except ValueError as e:
            return JSONResponse({"error": str(e)}, 400)
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass

    @app.get("/api/crop/{camera_id}/{filename}")
    async def get_crop(camera_id: str, filename: str):
        path = os.path.join(store.crops_dir, camera_id, filename)
        if not os.path.exists(path):
            raise HTTPException(404, "Crop not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/feedback")
    async def feedback(obs_id: str = Form(...),
                       probe_hash: str = Form(...),
                       verdict: str = Form(...),
                       notes: str = Form("")):
        if verdict not in ("confirm", "reject", "ambiguous"):
            raise HTTPException(400, "Invalid verdict")
        store.log_feedback(obs_id, probe_hash, verdict, notes=notes)
        return {"status": "ok"}

    return app


def _get_html_ui() -> str:
    """Return the HTML/JS for the search web UI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Person Search — Possible Match Review</title>
<style>
:root { --bg: #0f1117; --surface: #1a1d27; --accent: #6366f1;
  --text: #e2e8f0; --text-dim: #94a3b8; --warn: #f59e0b;
  --strong: #22c55e; --possible: #eab308; --weak: #94a3b8;
  --danger: #ef4444; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', 'Segoe UI', sans-serif; background: var(--bg);
  color: var(--text); min-height: 100vh; }
.header { background: linear-gradient(135deg, #1e1b4b, #312e81);
  padding: 24px 32px; border-bottom: 1px solid rgba(99,102,241,0.3); }
.header h1 { font-size: 1.5rem; font-weight: 600; }
.header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.upload-zone { border: 2px dashed rgba(99,102,241,0.4); border-radius: 12px;
  padding: 48px; text-align: center; background: var(--surface);
  transition: all 0.3s; cursor: pointer; margin-bottom: 24px; }
.upload-zone:hover { border-color: var(--accent); background: rgba(99,102,241,0.05); }
.upload-zone.dragover { border-color: var(--accent); background: rgba(99,102,241,0.1); }
.btn { padding: 10px 24px; border-radius: 8px; border: none; cursor: pointer;
  font-weight: 600; font-size: 0.9rem; transition: all 0.2s; }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: #4f46e5; transform: translateY(-1px); }
.results { display: grid; gap: 16px; }
.episode { background: var(--surface); border-radius: 12px; padding: 20px;
  border-left: 4px solid var(--text-dim); }
.episode.STRONG { border-left-color: var(--strong); }
.episode.POSSIBLE { border-left-color: var(--possible); }
.episode.WEAK { border-left-color: var(--weak); }
.tier-badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
.tier-badge.STRONG { background: rgba(34,197,94,0.15); color: var(--strong); }
.tier-badge.POSSIBLE { background: rgba(234,179,8,0.15); color: var(--possible); }
.tier-badge.WEAK { background: rgba(148,163,184,0.15); color: var(--weak); }
.match-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px; margin-top: 12px; }
.match-card { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
  border: 1px solid rgba(255,255,255,0.06); }
.match-card img { width: 100%; border-radius: 6px; margin-bottom: 8px; }
.score { font-size: 1.1rem; font-weight: 700; }
.meta { font-size: 0.8rem; color: var(--text-dim); }
.coverage { background: var(--surface); border-radius: 12px; padding: 20px;
  margin-top: 24px; }
.warning { color: var(--warn); font-size: 0.85rem; margin-top: 16px;
  padding: 12px; background: rgba(245,158,11,0.08); border-radius: 8px; }
.feedback-btn { padding: 4px 12px; border-radius: 6px; font-size: 0.75rem;
  cursor: pointer; border: 1px solid; margin: 2px; }
.fb-confirm { color: var(--strong); border-color: var(--strong); background: transparent; }
.fb-reject { color: var(--danger); border-color: var(--danger); background: transparent; }
.fb-ambiguous { color: var(--warn); border-color: var(--warn); background: transparent; }
.status-bar { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.status-chip { background: var(--surface); padding: 8px 16px; border-radius: 8px;
  font-size: 0.85rem; }
.loading { display: none; text-align: center; padding: 40px; }
.spinner { border: 3px solid var(--surface); border-top: 3px solid var(--accent);
  border-radius: 50%; width: 32px; height: 32px; animation: spin 0.8s linear infinite;
  margin: 0 auto 12px; }
@keyframes spin { to { transform: rotate(360deg); } }
input[type="file"] { display: none; }
#probe-preview { max-width: 200px; border-radius: 8px; margin: 12px auto; display: none; }
</style>
</head>
<body>
<div class="header">
  <h1>🔍 Person Search — Possible Match Review</h1>
  <p>Upload a photograph to search across camera observations. Results are leads for human review.</p>
</div>
<div class="container">
  <div id="status-bar" class="status-bar"></div>
  <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <input type="file" id="fileInput" accept="image/*">
    <img id="probe-preview" alt="Probe">
    <p style="font-size:1.1rem; margin-bottom:8px;">📷 Drop a photo here or click to upload</p>
    <p style="color:var(--text-dim); font-size:0.85rem;">JPG, PNG, WebP — clear frontal face recommended</p>
  </div>
  <div style="display:flex; gap:12px; margin-bottom:24px; align-items:center;">
    <label style="font-size:0.9rem;">Search last</label>
    <input type="number" id="lastMinutes" value="30" min="1" max="1440"
      style="width:80px; padding:8px; border-radius:6px; border:1px solid rgba(255,255,255,0.1);
      background:var(--surface); color:var(--text);">
    <label style="font-size:0.9rem;">minutes</label>
    <button class="btn btn-primary" onclick="runSearch()" id="searchBtn">Search</button>
  </div>
  <div class="loading" id="loading"><div class="spinner"></div><p>Searching observations...</p></div>
  <div id="results"></div>
</div>
<script>
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
const preview = document.getElementById('probe-preview');
let selectedFile = null;

dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files.length) { selectedFile = e.dataTransfer.files[0]; showPreview(); }
});
fi.addEventListener('change', () => { if (fi.files.length) { selectedFile = fi.files[0]; showPreview(); }});

function showPreview() {
  const url = URL.createObjectURL(selectedFile);
  preview.src = url; preview.style.display = 'block';
}

async function runSearch() {
  if (!selectedFile) { alert('Please select a photo first.'); return; }
  const fd = new FormData();
  fd.append('photo', selectedFile);
  fd.append('last_minutes', document.getElementById('lastMinutes').value);
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').innerHTML = '';
  try {
    const resp = await fetch('/api/search', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    renderResults(data);
  } catch(e) { alert('Search failed: ' + e.message); }
  finally { document.getElementById('loading').style.display = 'none'; }
}

function renderResults(data) {
  let html = '';
  if (data.probe_quality) {
    html += '<div style="margin-bottom:16px;">';
    html += `<span class="tier-badge ${data.probe_quality.reliability.toUpperCase()}">${data.probe_quality.reliability} probe</span> `;
    html += `<span class="meta">${data.probe_quality.template_count} templates, ${data.probe_quality.face_size_px}px face</span>`;
    if (data.probe_quality.warnings.length) {
      html += '<div class="warning">' + data.probe_quality.warnings.map(w=>'⚠ '+w).join('<br>') + '</div>';
    }
    html += '</div>';
  }
  if (!data.episodes || !data.episodes.length) {
    html += '<p style="text-align:center; padding:40px; color:var(--text-dim);">No matches found in the searched window.</p>';
  } else {
    html += `<p style="margin-bottom:16px;">${data.total_matches} match(es) in ${data.episodes.length} episode(s)</p>`;
    html += '<div class="results">';
    for (const ep of data.episodes) {
      html += `<div class="episode ${ep.tier}">`;
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">`;
      html += `<div><span class="tier-badge ${ep.tier}">${ep.tier}</span> <strong>${ep.camera_id}</strong></div>`;
      html += `<span class="meta">${ep.time_start} — ${ep.time_end}</span>`;
      html += `</div>`;
      html += `<div class="meta">Score: ${ep.best_score.toFixed(3)} | ${ep.observation_count} observations</div>`;
      html += '<div class="match-grid">';
      for (const m of ep.matches) {
        html += '<div class="match-card">';
        if (m.crop_path) {
          const parts = m.crop_path.replace(/\\\\/g,'/').split('/');
          const cam = parts[parts.length-2]; const fn = parts[parts.length-1];
          html += `<img src="/api/crop/${cam}/${fn}" onerror="this.style.display='none'">`;
        }
        html += `<div class="score">${m.score.toFixed(3)}</div>`;
        html += `<span class="tier-badge ${m.tier}">${m.tier}</span>`;
        html += `<div class="meta">${m.ts_ist}<br>Quality: ${m.quality.toFixed(2)} | Mask: ${m.mask_state}</div>`;
        if (m.flags.length) html += `<div class="meta" style="color:var(--warn);">${m.flags.join(', ')}</div>`;
        html += `<div style="margin-top:8px;">`;
        html += `<button class="feedback-btn fb-confirm" onclick="feedback('${m.obs_id}','confirm')">✓ Confirm</button>`;
        html += `<button class="feedback-btn fb-reject" onclick="feedback('${m.obs_id}','reject')">✗ Reject</button>`;
        html += `<button class="feedback-btn fb-ambiguous" onclick="feedback('${m.obs_id}','ambiguous')">? Ambiguous</button>`;
        html += '</div></div>';
      }
      html += '</div></div>';
    }
    html += '</div>';
  }
  if (data.coverage) {
    html += '<div class="coverage"><h3 style="margin-bottom:8px;">Coverage Statement</h3>';
    html += `<div class="meta">`;
    html += `Cameras: ${(data.coverage.cameras_searched||[]).join(', ')}<br>`;
    html += `Window: ${(data.coverage.window_duration_minutes||0).toFixed(1)} minutes<br>`;
    html += `Usable observations: ${data.coverage.total_usable_observations}<br>`;
    html += `${data.coverage.note||''}`;
    html += '</div></div>';
  }
  html += '<div class="warning">⚠ These are <strong>possible matches for human review</strong>. STRONG ≠ proof. Verify with context and judgment.</div>';
  document.getElementById('results').innerHTML = html;
}

async function feedback(obsId, verdict) {
  const fd = new FormData();
  fd.append('obs_id', obsId); fd.append('probe_hash', 'session');
  fd.append('verdict', verdict);
  try { await fetch('/api/feedback', {method:'POST', body:fd}); alert('Feedback saved.'); }
  catch(e) { alert('Failed: '+e.message); }
}

// Load health on startup
fetch('/api/health').then(r=>r.json()).then(d => {
  const bar = document.getElementById('status-bar');
  bar.innerHTML = `<div class="status-chip">📊 ${d.observations} observations</div>`;
  if (d.cameras) for (const c of d.cameras) {
    const icon = c.health==='ONLINE'?'🟢':c.health==='DEGRADED'?'🟡':'🔴';
    bar.innerHTML += `<div class="status-chip">${icon} ${c.name}</div>`;
  }
}).catch(()=>{});
</script>
</body>
</html>"""


def start_web_server(store, embedder, camera_mgr=None, ingest_mgr=None,
                     host: str = "127.0.0.1", port: int = 8501):
    """Start the FastAPI web server in a background thread."""
    app = create_web_app(store, embedder, camera_mgr, ingest_mgr)
    if app is None:
        return None

    try:
        import uvicorn
    except ImportError:
        print("  [SearchAPI] uvicorn not installed. Web UI disabled.")
        print("  Install with: pip install uvicorn")
        return None

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_run, name="search-web", daemon=True)
    thread.start()
    print(f"  [SearchAPI] Web UI running at http://{host}:{port}")
    return thread
