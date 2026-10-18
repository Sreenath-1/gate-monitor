@echo off
REM stop_sim.bat — Stop MediaMTX + ffmpeg
echo Stopping RTSP simulator...
taskkill /IM ffmpeg.exe /F 2>nul
taskkill /IM mediamtx.exe /F 2>nul
echo Done.
