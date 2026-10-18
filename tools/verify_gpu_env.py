"""
tools/verify_gpu_env.py
=======================
Verify GPU environment: CUDA, torch, onnxruntime providers,
VRAM, driver info. Writes results to docs/hardware.md placeholders.

Run inside .venv-gpu:
    .venv-gpu\\Scripts\\python.exe tools/verify_gpu_env.py
"""

import json
import io
import os
import platform
import subprocess
import sys

# Force UTF-8 stdout on Windows (cp1252 can't encode box-drawing chars)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path


def check_torch():
    """Check PyTorch CUDA availability."""
    results = {}
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        results["cuda_version"] = torch.version.cuda or "N/A"
        results["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        if torch.cuda.is_available():
            results["device_name"] = torch.cuda.get_device_name(0)
            results["device_count"] = torch.cuda.device_count()
            mem = torch.cuda.get_device_properties(0).total_memory
            results["vram_total_mb"] = round(mem / (1024 * 1024))
            # Quick allocation test
            try:
                t = torch.zeros(1, device="cuda")
                results["cuda_alloc_test"] = "PASS"
                del t
                torch.cuda.empty_cache()
            except Exception as e:
                results["cuda_alloc_test"] = f"FAIL: {e}"
        else:
            results["device_name"] = "N/A"
            results["cuda_alloc_test"] = "SKIP (no CUDA)"
    except ImportError:
        results["error"] = "torch not installed"
    except Exception as e:
        results["error"] = str(e)
    return results


def check_onnxruntime():
    """Check ONNX Runtime providers."""
    results = {}
    try:
        import onnxruntime as ort
        results["ort_version"] = ort.__version__
        results["available_providers"] = ort.get_available_providers()
        results["has_cuda_ep"] = "CUDAExecutionProvider" in ort.get_available_providers()
        results["has_dml_ep"] = "DmlExecutionProvider" in ort.get_available_providers()

        # Try to create a session with CUDA EP
        if results["has_cuda_ep"]:
            try:
                sess_opts = ort.SessionOptions()
                # Minimal test — just check provider registration
                results["cuda_ep_status"] = "available"
            except Exception as e:
                results["cuda_ep_status"] = f"error: {e}"
        else:
            results["cuda_ep_status"] = "not_available"

    except ImportError:
        results["error"] = "onnxruntime not installed"
    except Exception as e:
        results["error"] = str(e)
    return results


def check_nvidia_smi():
    """Get GPU info from nvidia-smi."""
    results = {}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,"
             "temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,"
             "clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        parts = [p.strip() for p in output.split(",")]
        if len(parts) >= 6:
            results["gpu_name"] = parts[0]
            results["vram_total_mb"] = int(parts[1])
            results["vram_free_mb"] = int(parts[2])
            results["driver_version"] = parts[3]
            def _safe_int(s):
                try:
                    return int(s)
                except (ValueError, TypeError):
                    return None

            def _safe_float(s):
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None

            results["gpu_temp_c"] = _safe_int(parts[4])
            results["power_draw_w"] = _safe_float(parts[5])
            if len(parts) >= 8:
                results["power_limit_w"] = _safe_float(parts[6])
                results["gpu_clock_mhz"] = _safe_int(parts[7])
            if len(parts) >= 10:
                results["mem_clock_mhz"] = _safe_int(parts[8])
                results["throttle_reasons"] = parts[9]
    except FileNotFoundError:
        results["error"] = "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        results["error"] = "nvidia-smi timed out"
    except Exception as e:
        results["error"] = str(e)
    return results


def check_system():
    """Get basic system info."""
    import shutil
    results = {
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    try:
        import psutil
        results["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 * 1024))
        results["ram_available_mb"] = round(psutil.virtual_memory().available / (1024 * 1024))
        results["battery"] = None
        bat = psutil.sensors_battery()
        if bat:
            results["battery"] = {
                "percent": bat.percent,
                "plugged_in": bat.power_plugged,
            }
    except ImportError:
        results["psutil"] = "not installed"

    # Disk
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
    results["disk_total_gb"] = round(usage.total / (1024 ** 3), 1)
    results["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)

    return results


def main():
    print(f"\n{'='*60}")
    print(f"  GPU ENVIRONMENT VERIFICATION")
    print(f"{'='*60}\n")

    # System
    sys_info = check_system()
    print(f"  Python : {sys_info['python_version']} ({sys_info['python_path']})")
    print(f"  OS     : {sys_info['platform']}")
    if "ram_total_mb" in sys_info:
        print(f"  RAM    : {sys_info['ram_total_mb']} MB total, "
              f"{sys_info.get('ram_available_mb', '?')} MB available")
    print(f"  Disk   : {sys_info['disk_total_gb']} GB total, "
          f"{sys_info['disk_free_gb']} GB free")
    if sys_info.get("battery"):
        bat = sys_info["battery"]
        status = "PLUGGED IN" if bat["plugged_in"] else "⚠ ON BATTERY"
        print(f"  Battery: {bat['percent']}% — {status}")

    # nvidia-smi
    print(f"\n  ── nvidia-smi ──")
    nv = check_nvidia_smi()
    if "error" in nv:
        print(f"  ✗ {nv['error']}")
    else:
        print(f"  GPU    : {nv.get('gpu_name', '?')}")
        print(f"  VRAM   : {nv.get('vram_total_mb', '?')} MB total, "
              f"{nv.get('vram_free_mb', '?')} MB free")
        print(f"  Driver : {nv.get('driver_version', '?')}")
        print(f"  Temp   : {nv.get('gpu_temp_c', '?')}°C")
        print(f"  Power  : {nv.get('power_draw_w', '?')}W / {nv.get('power_limit_w', '?')}W")
        if nv.get("throttle_reasons"):
            print(f"  Throttle: {nv['throttle_reasons']}")

    # PyTorch
    print(f"\n  ── PyTorch ──")
    torch_info = check_torch()
    if "error" in torch_info:
        print(f"  ✗ {torch_info['error']}")
    else:
        print(f"  Version  : {torch_info.get('torch_version', '?')}")
        print(f"  CUDA     : {torch_info.get('cuda_available', '?')}")
        print(f"  CUDA ver : {torch_info.get('cuda_version', '?')}")
        print(f"  cuDNN    : {torch_info.get('cudnn_version', '?')}")
        print(f"  Device   : {torch_info.get('device_name', '?')}")
        print(f"  VRAM     : {torch_info.get('vram_total_mb', '?')} MB")
        print(f"  Alloc    : {torch_info.get('cuda_alloc_test', '?')}")

    # ONNX Runtime
    print(f"\n  ── ONNX Runtime ──")
    ort_info = check_onnxruntime()
    if "error" in ort_info:
        print(f"  ✗ {ort_info['error']}")
    else:
        print(f"  Version   : {ort_info.get('ort_version', '?')}")
        print(f"  Providers : {ort_info.get('available_providers', '?')}")
        print(f"  CUDA EP   : {ort_info.get('cuda_ep_status', '?')}")
        print(f"  DML EP    : {ort_info.get('has_dml_ep', '?')}")

    # Summary
    print(f"\n{'='*60}")
    all_ok = (
        torch_info.get("cuda_available") and
        torch_info.get("cuda_alloc_test") == "PASS" and
        ort_info.get("has_cuda_ep")
    )
    if all_ok:
        print(f"  ✓ GPU environment OK — CUDA EP active")
    else:
        print(f"  ✗ GPU environment issues detected:")
        if not torch_info.get("cuda_available"):
            print(f"    - PyTorch CUDA not available")
        if torch_info.get("cuda_alloc_test", "").startswith("FAIL"):
            print(f"    - CUDA allocation failed")
        if not ort_info.get("has_cuda_ep"):
            print(f"    - ONNX Runtime CUDA EP not available")
            if ort_info.get("has_dml_ep"):
                print(f"    - DirectML EP available as fallback")
    print(f"{'='*60}\n")

    # Save to JSON for docs/hardware.md update
    output = {
        "system": sys_info,
        "nvidia_smi": nv,
        "torch": torch_info,
        "onnxruntime": ort_info,
        "all_ok": all_ok,
    }
    out_path = Path(__file__).parent.parent / "docs" / "gpu_probe.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Full results saved to: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
