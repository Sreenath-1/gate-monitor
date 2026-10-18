# Gate Monitor

Camera-based gate monitoring for RTSP CCTV. One codebase with several modes, selected with `main.py --mode <mode>`:

| Mode | Purpose | Status |
|---|---|---|
| `attendance` | Detects and tracks people at a gate, identifies enrolled people by face, logs in/out events with IST timestamps | existing |
| `search` | Web UI (FastAPI) to search stored person observations | paused |
| `sources-check` | Checks that configured camera sources are reachable | existing |
| `vehicles` | Detects vehicles, reads Indian number plates, stores events, serves a dashboard | development, Phase 1 |

**Honest status:** the vehicle pipeline has unit tests and speed benchmarks on synthetic frames only. It has not been run on real camera footage and no accuracy has been measured. No camera is installed yet.

## Shared building blocks

- `camera_manager.py`, `stream.py`: multi-camera RTSP capture with reconnect and health monitoring, configured in `cameras.yaml` (an example is provided; real URLs come from environment variables).
- `zone.py`: virtual line crossing and direction.
- `logger.py`, `config.py`: SQLite event logging with IST timestamps and CLI/config handling.
- `tools/`: calibration, camera probing, benchmarks, evaluation and test-stream helpers.
- `scripts/`: offline pipeline tests and open-set rejection tests (standalone, not run by pytest).

## Person features (attendance, search)

Person detection with YOLO (Ultralytics) and BoT-SORT tracking, face identification with InsightFace, a FastAPI web UI on 127.0.0.1. Face enrollment data and attendance reports are personal data and are not in this repository (see `known_faces/README.md`).

## Vehicle ANPR

Pipeline: RTSP capture, vehicle detection (RF-DETR Nano), tracking (ByteTrack), plate detection on vehicle crops (YOLOv9-t), plate OCR (fast-plate-ocr), Indian plate validation (standard, BH series, legacy formats, state codes, positional correction that is always flagged), cross-frame voting, SQLite event store, FastAPI dashboard with search, filters, CSV export and operator review.

Modules: `vehicle_detector.py`, `vehicle_tracker.py`, `plate_reader.py`, `vehicle_pipeline.py`, `validators/indian_plate.py`, `event_store.py`, `dashboard.py`. Design decisions are in `docs/decisions.md`.

Measured so far (synthetic 1920x1080 frames, RTX 4050 Laptop 6 GB): RF-DETR Nano 29.6 FPS using 173 MB, plate detection plus OCR about 25 ms per vehicle crop. These show compute speed only, not accuracy.

Known limitations: two-line and hand-painted plates are an expected weak point of the global OCR model; night and infrared glare are untested; the development laptop is not suitable for 24/7 production use.

## Setup

Windows, Python 3.11. Model weights, face data, databases, videos and credentials are not stored in the repository.

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -r requirements-vehicles.txt
    copy .env.example .env
    copy cameras.yaml.example cameras.yaml

Vehicle tests (need no camera or model weights):

    python -m pytest tests/ -q

Standalone integration tests (need model weights or local data):

    python scripts/test_pipeline_offline.py
    python scripts/test_open_set_rejection.py

## Repository history

Earlier history is not published. This repository starts from a cleaned snapshot of the project; camera addresses and credentials are placeholders (set `RTSP_URL` in `.env`). Later work is reviewed through pull requests.

## Licensing

All rights reserved. Third-party components keep their own licences; see `LICENSING.md`. Ultralytics YOLO is AGPL-3.0 and InsightFace weights are non-commercial research only, so review both before any commercial use.
