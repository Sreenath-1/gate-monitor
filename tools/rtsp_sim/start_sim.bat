@echo off
REM start_sim.bat — Start MediaMTX + ffmpeg RTSP loop
REM Usage: start_sim.bat <video_file.mp4>
REM Stream available at: rtsp://127.0.0.1:8554/sim

if "%~1"=="" (
    echo Usage: start_sim.bat ^<video_file.mp4^>
    echo.
    echo Example: start_sim.bat data\clips\test_clip.mp4
    exit /b 1
)

set VIDEO_FILE=%~1
set SCRIPT_DIR=%~dp0

REM Check prerequisites
where mediamtx >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "%SCRIPT_DIR%mediamtx.exe" (
        echo ERROR: mediamtx.exe not found on PATH or in %SCRIPT_DIR%
        echo Download from: https://github.com/bluenviron/mediamtx/releases
        exit /b 1
    )
    set MEDIAMTX_EXE=%SCRIPT_DIR%mediamtx.exe
) else (
    set MEDIAMTX_EXE=mediamtx
)

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "%SCRIPT_DIR%ffmpeg.exe" (
        echo ERROR: ffmpeg.exe not found on PATH or in %SCRIPT_DIR%
        echo Download from: https://github.com/BtbN/FFmpeg-Builds/releases
        exit /b 1
    )
    set FFMPEG_EXE=%SCRIPT_DIR%ffmpeg.exe
) else (
    set FFMPEG_EXE=ffmpeg
)

if not exist "%VIDEO_FILE%" (
    echo ERROR: Video file not found: %VIDEO_FILE%
    exit /b 1
)

echo ============================================================
echo   RTSP Simulator
echo ============================================================
echo   Video  : %VIDEO_FILE%
echo   Stream : rtsp://127.0.0.1:8554/sim
echo   Press Ctrl+C to stop
echo ============================================================
echo.

REM Start MediaMTX in background
start "MediaMTX" /MIN "%MEDIAMTX_EXE%" "%SCRIPT_DIR%mediamtx.yml"
timeout /t 2 /nobreak >nul

REM Loop ffmpeg — -stream_loop -1 loops forever
:loop
echo [%TIME%] Starting ffmpeg publish loop...
"%FFMPEG_EXE%" -re -stream_loop -1 -i "%VIDEO_FILE%" -c copy -f rtsp rtsp://127.0.0.1:8554/sim
echo [%TIME%] ffmpeg exited, restarting in 2s...
timeout /t 2 /nobreak >nul
goto loop
