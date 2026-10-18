# Hardware Profile — Vehicle ANPR Dev Machine

> **Probed**: 2026-10-15 IST | **Machine**: Acer Aspire 7 A715-79G (dev + pilot only)

## Hardware

| Component | Value |
|-----------|-------|
| **CPU** | Intel Core 7 240H (10 cores / 16 threads) |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **VRAM** | 6,141 MiB (6 GB) |
| **iGPU** | Intel UHD (hybrid graphics) |
| **RAM** | 16,072 MB DDR4 — **single SODIMM, single-channel** |
| **SSD** | 512 GB NVMe |
| **Cooling** | 2-fan, no vapor chamber |
| **Adapter** | 150 W |

## Software

| Component | Value |
|-----------|-------|
| **OS** | Windows 11 Home, Build 26200 |
| **NVIDIA Driver** | 576.93 |
| **CUDA (driver)** | 12.9 |
| **Python (embedded)** | 3.11.9 at `.python/python.exe` (CPU torch, no venv) |
| **Python (venv)** | 3.11.9 via virtualenv at `.venv-gpu/` |

## GPU Verification

> Probed: 2026-10-15 17:46 IST — **ALL PASS**

| Check | Result |
|-------|--------|
| `torch.cuda.is_available()` | ✅ `True` |
| `torch.cuda.get_device_name(0)` | NVIDIA GeForce RTX 4050 Laptop GPU |
| `torch.version.cuda` | 12.4 |
| `torch.__version__` | 2.6.0+cu124 |
| `torch.backends.cudnn.version()` | 90100 (cuDNN 9.1.0) |
| CUDA alloc test | ✅ PASS |
| `onnxruntime.__version__` | 1.21.1 |
| `onnxruntime.get_available_providers()` | TensorrtExecutionProvider, **CUDAExecutionProvider**, CPUExecutionProvider |
| CUDA EP active | ✅ Yes |
| VRAM total / free (idle) | 6140 MB / 5921 MB |
| GPU temp (idle) | 46°C |

## Engineering Budgets

| Resource | Budget | Rationale |
|----------|--------|-----------|
| **VRAM** | ≤ 4 GB | 6 GB total, ~2 GB for OS + display driver |
| **System RAM** | ≤ 4 GB RSS | 16 GB shared with IDE, tools, browser |
| **Disk (data/)** | ≤ 80 GB | Refuse clip recording below 20 GB free |
| **Sustained FPS** | ≥ 12 FPS | Measured after 30 min thermal soak |
| **p95 event latency** | ≤ 2 s | From vehicle exit to plate emitted |

## Thermal Notes

- No vapor chamber — sustained GPU load will thermal-throttle
- Adaptive load-shedding enabled at GPU temp > 80°C (configurable)
- All benchmarks must report sustained FPS (30 min), not peak
- Always run plugged in, Windows power mode "Best performance"
- Lid-close action: "Do nothing"
- Optional cooling pad recommended for 24/7 pilot
