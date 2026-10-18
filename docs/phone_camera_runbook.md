# Phone-as-Camera Runbook

Quick guide to use an Android phone as an RTSP test camera for the vehicle
ANPR pipeline.

## Setup (one-time)

### 1. Install an RTSP camera app on your phone

**Recommended: IP Webcam** (Pavel Khlebovich, free)
- Play Store: search "IP Webcam"
- Alternatives: RTSP Camera Server, CameraFi

### 2. Connect phone to the same network as the laptop

- **Best**: both on the same Wi-Fi router
- **Better**: phone hotspot → laptop connects (avoids router latency)
- Phone and laptop must be able to reach each other (ping test)

### 3. Start the camera stream

1. Open IP Webcam
2. Settings:
   - **Video resolution**: 1920×1080 (Full HD)
   - **Video quality**: 80-100%
   - **Audio**: OFF (saves bandwidth)
   - **FPS limit**: 25
3. Scroll down → "Start server"
4. Note the URL shown (e.g. `http://CAMERA_IP:8080`)

### 4. Find the RTSP URL

IP Webcam exposes:
- HTTP MJPEG: `http://<phone-ip>:8080/video`
- RTSP: `rtsp://<phone-ip>:8080/h264_opus.sdp`

Test with VLC: Media → Open Network Stream → paste RTSP URL.

### 5. Configure gate_monitor

Add to `.env` (you do this, I never edit .env):
```
PHONE_RTSP_URL=rtsp://CAMERA_IP:8080/h264_opus.sdp
```

In `cameras.yaml`, the `phone_test` entry is already configured to use
`${PHONE_RTSP_URL}`.

## Recording test clips

```bash
.venv-gpu\Scripts\python.exe tools/record_clips.py --camera phone_test --duration 2 --count 3
```

This records 2-minute clips into `data/clips/`.

## Test positioning

For ANPR testing, position the phone to simulate a gate camera:

1. Mount phone on a tripod or tape it to a wall at **1.5-2m height**
2. Angle **~30° downward** toward where vehicles pass
3. Vehicles should fill **~30-60% of the frame width** when passing
4. Licence plates should be **clearly readable** to the human eye
5. Record clips with vehicles entering AND exiting

## What to record

| Scenario | Duration | Notes |
|----------|----------|-------|
| Daytime, cars passing | 5 min | Basic functionality |
| Daytime, two-wheelers | 5 min | Two-line plate format |
| Evening / low light | 5 min | Quality degradation |
| Mixed vehicles | 10 min | Realistic traffic |

## After recording

1. Watch each clip, note vehicles + plates (fill `data/labels.csv`)
2. Run the plate size probe:
   ```bash
   .venv-gpu\Scripts\python.exe tools/probe_plate_size.py --source data/clips/
   ```
3. Review annotated frames in `data/review/plate_probe/`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Can't connect | Check firewall, try `ping <phone-ip>` |
| Black frames | IP Webcam may have paused — unlock phone |
| Low FPS | Reduce resolution to 1280×720, disable audio |
| Lag / buffering | Use `rtsp_transport;tcp` (already set in stream.py) |
| Phone gets hot | Lower resolution, point a fan at it |
| Wi-Fi drops | Use phone hotspot instead of router Wi-Fi |
