# tools/rtsp_sim — Simulated RTSP Server
#
# Loops a local MP4 file as a live RTSP stream using MediaMTX + ffmpeg.
# Useful for testing the pipeline without a physical camera.
#
# PREREQUISITES (you install manually — never curl|sh):
#   1. MediaMTX: download from https://github.com/bluenviron/mediamtx/releases
#      Get the Windows amd64 ZIP, extract mediamtx.exe into this directory.
#   2. ffmpeg:   download from https://github.com/BtbN/FFmpeg-Builds/releases
#      Get ffmpeg-master-latest-win64-gpl.zip, put ffmpeg.exe on PATH
#      or into this directory.
#
# USAGE:
#   # Start the RTSP server:
#   tools\rtsp_sim\start_sim.bat data\clips\test_clip.mp4
#
#   # The stream will be available at:
#   #   rtsp://127.0.0.1:8554/sim
#
#   # Set in .env:
#   #   PHONE_RTSP_URL=rtsp://127.0.0.1:8554/sim
#
# FILES:
#   start_sim.bat     — Launches MediaMTX + ffmpeg loop
#   stop_sim.bat      — Kills both processes
#   mediamtx.yml      — Minimal MediaMTX config (RTSP only, no auth)
#   README.md         — This file (setup instructions)
#
# FAULT INJECTION:
#   # Kill publisher (ffmpeg) to test reconnect:
#   taskkill /IM ffmpeg.exe /F
#   # Wait N seconds, then restart:
#   tools\rtsp_sim\start_sim.bat data\clips\test_clip.mp4
#
#   # Simulate black frames:
#   ffmpeg -f lavfi -i color=black:s=1920x1080:r=25 -t 10 -f rtsp rtsp://127.0.0.1:8554/sim
#
#   # Simulate packet loss (requires tc/clumsy on Windows):
#   # Use Clumsy (https://jagt.github.io/clumsy/) with Drop=5%
