"""
tools/probe_camera.py
=====================
RTSP URL discovery tool for DVR/IP cameras.

Tries common RTSP URL templates for major DVR brands against a target IP
and reports which URLs successfully open + their resolution/FPS/codec.

Usage:
    python tools/probe_camera.py CAMERA_IP
    python tools/probe_camera.py CAMERA_IP --user admin --pass admin
    python tools/probe_camera.py CAMERA_IP --port 554 --timeout 10

Supported brands: Hikvision, Dahua/CP Plus, VisionX, Generic ONVIF.
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2


# ─────────────────────────────────────────────────────────────────────────────
#  RTSP URL TEMPLATES BY BRAND
# ─────────────────────────────────────────────────────────────────────────────
#  {user}, {pass}, {ip}, {port}, {channel} are substituted at runtime.
#  "main" and "sub" indicate mainstream (hi-res) and substream (low-res).

TEMPLATES = [
    # ── Generic / VisionX ──────────────────────────────────────────────
    {
        "brand": "Generic/VisionX",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/live/0/MAIN",
    },
    {
        "brand": "Generic/VisionX",
        "quality": "sub",
        "template": "rtsp://{auth}{ip}:{port}/live/0/SUB",
    },

    # ── Hikvision ──────────────────────────────────────────────────────
    {
        "brand": "Hikvision",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/Streaming/Channels/{channel}01",
    },
    {
        "brand": "Hikvision",
        "quality": "sub",
        "template": "rtsp://{auth}{ip}:{port}/Streaming/Channels/{channel}02",
    },
    {
        "brand": "Hikvision (alt)",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/ISAPI/Streaming/Channels/{channel}01",
    },

    # ── Dahua / CP Plus ────────────────────────────────────────────────
    {
        "brand": "Dahua/CP Plus",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
    },
    {
        "brand": "Dahua/CP Plus",
        "quality": "sub",
        "template": "rtsp://{auth}{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
    },

    # ── ONVIF Generic ──────────────────────────────────────────────────
    {
        "brand": "ONVIF Generic",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/onvif1",
    },
    {
        "brand": "ONVIF Generic",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/stream1",
    },
    {
        "brand": "ONVIF Generic",
        "quality": "sub",
        "template": "rtsp://{auth}{ip}:{port}/stream2",
    },

    # ── Axis ───────────────────────────────────────────────────────────
    {
        "brand": "Axis",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/axis-media/media.amp",
    },

    # ── Reolink ────────────────────────────────────────────────────────
    {
        "brand": "Reolink",
        "quality": "main",
        "template": "rtsp://{auth}{ip}:{port}/h264Preview_01_main",
    },
    {
        "brand": "Reolink",
        "quality": "sub",
        "template": "rtsp://{auth}{ip}:{port}/h264Preview_01_sub",
    },
]


def probe_url(url: str, timeout: int = 10) -> dict:
    """Try to open an RTSP URL and read a few frames.

    Args:
        url: Full RTSP URL to test.
        timeout: Max seconds to wait for connection.

    Returns:
        Dict with success, resolution, fps, codec, read_time, frames_read.
    """
    result = {
        "url": url,
        "success": False,
        "resolution": "",
        "fps": 0.0,
        "frames_read": 0,
        "read_time_ms": 0,
        "error": None,
    }

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Wait for connection with timeout
        if not cap.isOpened():
            result["error"] = "Failed to open"
            return result

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        result["resolution"] = f"{w}x{h}"
        result["fps"] = fps

        # Try to read 3 frames
        t0 = time.time()
        for _ in range(3):
            if time.time() - t0 > timeout:
                result["error"] = "Timeout reading frames"
                break
            ret, frame = cap.read()
            if ret:
                result["frames_read"] += 1
            else:
                break

        result["read_time_ms"] = int((time.time() - t0) * 1000)
        result["success"] = result["frames_read"] >= 1
        cap.release()

    except Exception as e:
        result["error"] = str(e)

    return result


def run_probe(ip: str, user: str = "", password: str = "",
              port: int = 554, channel: int = 1, timeout: int = 10):
    """Try all RTSP URL templates against a target IP.

    Args:
        ip: Target camera/DVR IP address.
        user: RTSP username (empty = no auth).
        password: RTSP password.
        port: RTSP port (default 554).
        channel: DVR channel number (default 1).
        timeout: Connection timeout per URL.
    """
    auth = f"{user}:{password}@" if user else ""

    print(f"\n{'='*70}")
    print(f"  RTSP CAMERA PROBE")
    print(f"{'='*70}")
    print(f"  Target : {ip}:{port}")
    print(f"  Auth   : {'yes' if user else 'none'}")
    print(f"  Channel: {channel}")
    print(f"  Timeout: {timeout}s per URL")
    print(f"  Testing {len(TEMPLATES)} URL templates...\n")

    found = []
    failed = []

    for tmpl in TEMPLATES:
        url = tmpl["template"].format(
            auth=auth, ip=ip, port=port, channel=channel
        )

        # Display URL with credentials masked
        display_url = url
        if password:
            display_url = url.replace(password, "****")

        sys.stdout.write(f"  Testing {tmpl['brand']:20s} {tmpl['quality']:4s} ... ")
        sys.stdout.flush()

        result = probe_url(url, timeout)

        if result["success"]:
            print(f"✓ {result['resolution']} @ {result['fps']:.0f}fps "
                  f"({result['frames_read']} frames in {result['read_time_ms']}ms)")
            found.append({**tmpl, **result})
        else:
            err = result.get("error", "unknown")
            print(f"✗ ({err})")
            failed.append({**tmpl, **result})

    # Summary
    print(f"\n{'='*70}")
    if found:
        print(f"  FOUND {len(found)} working URL(s):\n")
        for r in found:
            url = r["url"]
            if password:
                url = url.replace(password, "****")
            print(f"  [{r['brand']}] {r['quality']}stream")
            print(f"    URL : {url}")
            print(f"    Res : {r['resolution']} @ {r['fps']:.0f}fps\n")

        # Suggest cameras.yaml config
        print(f"  Suggested .env additions:")
        main_urls = [r for r in found if r["quality"] == "main"]
        sub_urls = [r for r in found if r["quality"] == "sub"]
        if main_urls:
            url = main_urls[0]["url"]
            if password:
                url = url.replace(password, "****")
            print(f"    CAMERA_CCTV_GATE_HIRES_URL={url}")
        if sub_urls:
            url = sub_urls[0]["url"]
            if password:
                url = url.replace(password, "****")
            print(f"    CAMERA_CCTV_GATE_URL={url}")
        elif main_urls:
            url = main_urls[0]["url"]
            if password:
                url = url.replace(password, "****")
            print(f"    CAMERA_CCTV_GATE_URL={url}")
    else:
        print(f"  No working URLs found.")
        print(f"  Try:")
        print(f"    - Different port (common: 554, 8554, 10554)")
        print(f"    - Check credentials (--user admin --pass admin)")
        print(f"    - Ensure camera/DVR is on the same network")
        print(f"    - Check firewall / router settings")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a camera/DVR IP for working RTSP URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/probe_camera.py CAMERA_IP
  python tools/probe_camera.py CAMERA_IP --user admin --pass admin123
  python tools/probe_camera.py CAMERA_IP --port 554 --channel 2
        """,
    )
    parser.add_argument("ip", help="Target camera/DVR IP address")
    parser.add_argument("--user", default="", help="RTSP username")
    parser.add_argument("--pass", dest="password", default="",
                        help="RTSP password")
    parser.add_argument("--port", type=int, default=554,
                        help="RTSP port (default: 554)")
    parser.add_argument("--channel", type=int, default=1,
                        help="DVR channel number (default: 1)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout per URL in seconds (default: 10)")

    args = parser.parse_args()
    run_probe(args.ip, args.user, args.password,
              args.port, args.channel, args.timeout)


if __name__ == "__main__":
    main()
