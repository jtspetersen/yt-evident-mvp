#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evident Video Fact Checker - Interactive Setup Wizard

Detects system hardware, recommends models, configures Docker services,
and downloads models for a complete one-command setup experience.
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import subprocess
import shutil
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    sys.exit(1)


# ----------------------------
# Configuration
# ----------------------------

MODEL_RECOMMENDATION_MATRIX = [
    # (min_vram_gb, min_ram_gb, profile_name, extract, verify, write)
    (24, 32, "High-end GPU (24GB+ VRAM)", "qwen3:8b", "qwen3:30b", "gemma3:27b"),
    (16, 32, "Mid-range GPU (16GB VRAM)", "qwen3:8b", "qwen3:14b", "gemma3:12b"),
    (12, 32, "Entry GPU (12GB VRAM)", "qwen3:8b", "qwen3:8b", "gemma3:9b"),
    (0, 64, "High-end CPU (64GB+ RAM)", "qwen3:8b", "qwen3:14b", "gemma3:9b"),
    (0, 32, "Mid-range CPU (32GB+ RAM)", "qwen3:8b", "qwen3:8b", "llama3:8b"),
    (0, 16, "Minimum (16GB RAM)", "phi3:mini", "phi3:mini", "phi3:mini"),
]


# ----------------------------
# Utilities
# ----------------------------

def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(text: str) -> None:
    """Print a step message."""
    print(f"→ {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"✗ ERROR: {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"⚠ WARNING: {text}")


def run_command(cmd: list, check: bool = True, capture: bool = False) -> Optional[str]:
    """
    Run a shell command.
    Returns stdout if capture=True, otherwise None.
    """
    try:
        if capture:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return None
    except FileNotFoundError:
        if check:
            raise
        return None


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{default_str}] ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


# ----------------------------
# Phase 1: Prerequisites Check
# ----------------------------

def _install_ffmpeg() -> bool:
    """Attempt to install FFmpeg automatically. Returns True if installed."""
    import platform
    system = platform.system()

    if system == "Windows":
        # Try winget (Windows Package Manager)
        print_step("Attempting to install FFmpeg via winget...")
        try:
            result = subprocess.run(
                ["winget", "install", "--id", "Gyan.FFmpeg",
                 "--accept-source-agreements", "--accept-package-agreements"],
                capture_output=False, text=True, timeout=300,
            )
            if result.returncode == 0:
                print_success("FFmpeg installed via winget")
                print_warning("Restart your terminal for PATH to take effect")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        print_warning("winget not available. Install FFmpeg manually:")
        print_warning("  winget install Gyan.FFmpeg")
        print_warning("  -- or download from https://ffmpeg.org/download.html")
        return False

    elif system == "Linux":
        # Try apt (Debian/Ubuntu)
        print_step("Attempting to install FFmpeg via apt...")
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"],
                           check=True, timeout=120)
            print_success("FFmpeg installed via apt")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return False

    elif system == "Darwin":
        # Try brew (macOS)
        print_step("Attempting to install FFmpeg via brew...")
        try:
            subprocess.run(["brew", "install", "ffmpeg"], check=True, timeout=300)
            print_success("FFmpeg installed via brew")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return False

    return False


def check_prerequisites() -> Dict[str, bool]:
    """Check if required tools are installed."""
    print_header("Phase 1: Prerequisites Check")

    results = {}

    # Docker
    print_step("Checking Docker...")
    try:
        version = run_command(["docker", "--version"], capture=True)
        print_success(f"Docker installed: {version}")
        results["docker"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker not installed or not in PATH")
        results["docker"] = False

    # Docker Compose
    print_step("Checking Docker Compose...")
    try:
        version = run_command(["docker", "compose", "version"], capture=True)
        print_success(f"Docker Compose installed: {version}")
        results["docker_compose"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Docker Compose v2 not installed")
        results["docker_compose"] = False

    # Python
    print_step("Checking Python version...")
    if sys.version_info >= (3, 11):
        print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} OK")
        results["python"] = True
    else:
        print_error(f"Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        results["python"] = False

    # FFmpeg (required for YouTube transcript Whisper fallback)
    print_step("Checking FFmpeg...")
    try:
        version = run_command(["ffmpeg", "-version"], capture=True)
        version_line = version.split('\n')[0] if version else "unknown"
        print_success(f"FFmpeg installed: {version_line}")
        results["ffmpeg"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("FFmpeg not found (needed for YouTube Whisper fallback)")
        if _install_ffmpeg():
            results["ffmpeg"] = True
        else:
            print_warning("FFmpeg not installed. YouTube Whisper fallback will not work.")
            print_warning("Caption-based transcripts will still work without it.")
            results["ffmpeg"] = False

    return results


# ----------------------------
# Phase 2: Hardware Detection
# ----------------------------

def detect_nvidia_gpu() -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Detect NVIDIA GPU and VRAM.
    Returns: (has_gpu, gpu_name, vram_gb)
    """
    print_step("Detecting NVIDIA GPU...")
    try:
        output = run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture=True,
            check=False
        )
        if output:
            parts = output.split(",")
            gpu_name = parts[0].strip()
            vram_mb = float(parts[1].strip())
            vram_gb = vram_mb / 1024
            print_success(f"NVIDIA GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return True, gpu_name, vram_gb
    except Exception:
        print_step("No NVIDIA GPU detected (nvidia-smi not available)")

    return False, None, None


def detect_amd_gpu_windows() -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Detect AMD GPU on Windows using PowerShell/wmic.
    Returns: (has_gpu, gpu_name, vram_gb)
    """
    try:
        import platform
        if platform.system() != "Windows":
            return False, None, None

        # Query GPU info using PowerShell (more reliable in Git Bash)
        output = run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_VideoController | Select-Object Name,AdapterRAM | Format-List"],
            capture=True,
            check=False
        )

        if output:
            # Parse PowerShell Format-List output
            gpu_name = None
            vram_bytes = None

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith("Name") and ("Radeon" in line or "AMD" in line):
                    # Extract GPU name after the colon
                    gpu_name = line.split(":", 1)[1].strip() if ":" in line else None
                elif line.startswith("AdapterRAM") and gpu_name:
                    # Extract VRAM bytes after the colon
                    try:
                        vram_str = line.split(":", 1)[1].strip() if ":" in line else ""
                        vram_bytes = int(vram_str) if vram_str else None
                    except ValueError:
                        pass

                    # If we have both name and VRAM for an AMD GPU, process it
                    if gpu_name and (vram_bytes or True):  # Accept even if VRAM not detected
                        # Convert VRAM from bytes to GB
                        vram_gb = None
                        if vram_bytes and vram_bytes > 0:
                            vram_gb = vram_bytes / (1024 ** 3)

                        # WMI AdapterRAM is 32-bit, can only report up to ~4GB
                        # Use lookup table for known high-VRAM GPUs
                        # Override if detected VRAM seems too low for the model
                        known_vram = None
                        if "7900 XTX" in gpu_name or "7900XTX" in gpu_name.replace(" ", ""):
                            known_vram = 24.0
                        elif "7900 XT" in gpu_name and "XTX" not in gpu_name:
                            known_vram = 20.0
                        elif "7800 XT" in gpu_name:
                            known_vram = 16.0
                        elif "7700 XT" in gpu_name:
                            known_vram = 12.0
                        elif "7600" in gpu_name:
                            known_vram = 8.0
                        elif "6950 XT" in gpu_name:
                            known_vram = 16.0
                        elif "6900 XT" in gpu_name:
                            known_vram = 16.0
                        elif "6800 XT" in gpu_name:
                            known_vram = 16.0
                        elif "6800" in gpu_name and "XT" not in gpu_name:
                            known_vram = 16.0
                        elif "6750 XT" in gpu_name:
                            known_vram = 12.0
                        elif "6700 XT" in gpu_name:
                            known_vram = 12.0

                        # Use known VRAM if detected is missing or suspiciously low
                        if known_vram and (not vram_gb or vram_gb < known_vram * 0.5):
                            vram_gb = known_vram

                        return True, gpu_name, vram_gb or 0.0
    except Exception:
        pass

    return False, None, None


def check_ollama_using_gpu() -> Tuple[bool, Optional[str]]:
    """
    Check if native Ollama is running and using GPU.
    Returns: (using_gpu, processor_info)
    """
    try:
        import urllib.request
        import json

        # Check if Ollama is running
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = data.get("models", [])
        except Exception:
            return False, None

        if not models:
            return False, None

        # Try to load a model and check if GPU is used
        # First check if any model is already loaded
        output = run_command(["ollama", "ps"], capture=True, check=False)
        if output and "GPU" in output:
            # Model already loaded with GPU
            return True, "GPU"

        # Try loading the smallest available model
        smallest_model = min(models, key=lambda m: m.get("size", float("inf")))
        model_name = smallest_model.get("name")
        if not model_name:
            return False, None

        # Quick inference to load model
        try:
            payload = json.dumps({
                "model": model_name,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1}
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                pass  # Just need to trigger model load
        except Exception:
            pass

        # Now check processor
        output = run_command(["ollama", "ps"], capture=True, check=False)
        if output:
            for line in output.split('\n'):
                if "GPU" in line:
                    # Extract processor info (e.g., "100% GPU" or "50% GPU/50% CPU")
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "GPU" in part or (i > 0 and "GPU" in parts[i-1] if i > 0 else False):
                            # Find the percentage
                            for p in parts:
                                if "%" in p and "GPU" in line:
                                    return True, line
                    return True, "GPU"

        return False, None
    except Exception:
        return False, None


def detect_amd_gpu() -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Detect AMD GPU with ROCm and VRAM.
    Returns: (has_gpu, gpu_name, vram_gb)
    """
    print_step("Detecting AMD GPU with ROCm...")

    # Try rocm-smi first (Linux/Windows with ROCm)
    try:
        output = run_command(
            ["rocm-smi", "--showproductname"],
            capture=True,
            check=False
        )
        if output and "GPU" in output:
            gpu_name = None
            for line in output.split('\n'):
                if "Card series:" in line or "Card model:" in line:
                    gpu_name = line.split(":")[-1].strip()
                    break

            if not gpu_name:
                # Try to extract from general output
                for line in output.split('\n'):
                    if "Radeon" in line or "AMD" in line:
                        gpu_name = line.strip()
                        break

            # Try to get VRAM info
            vram_gb = None
            try:
                vram_output = run_command(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture=True,
                    check=False
                )
                if vram_output:
                    # Parse VRAM size from output
                    for line in vram_output.split('\n'):
                        if "Total" in line or "VRAM Total" in line:
                            # Extract number (usually in MB or GB)
                            import re
                            match = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB|MiB|GiB)', line, re.IGNORECASE)
                            if match:
                                size = float(match.group(1))
                                unit = match.group(2).upper()
                                if 'MB' in unit or 'MIB' in unit:
                                    vram_gb = size / 1024
                                else:
                                    vram_gb = size
                                break
            except Exception:
                pass

            # Default VRAM estimate for known cards if detection fails
            if not vram_gb and gpu_name:
                if "7900 XTX" in gpu_name or "7900XTX" in gpu_name.replace(" ", ""):
                    vram_gb = 24.0
                elif "7900 XT" in gpu_name:
                    vram_gb = 20.0
                elif "7800 XT" in gpu_name:
                    vram_gb = 16.0

            if gpu_name:
                vram_str = f" ({vram_gb:.1f} GB VRAM)" if vram_gb else ""
                print_success(f"AMD GPU detected: {gpu_name}{vram_str}")
                return True, gpu_name, vram_gb or 0.0
    except Exception:
        pass

    # Fallback: Check for ROCm installation on Windows
    try:
        import platform
        if platform.system() == "Windows":
            # Check if HIP/ROCm is installed
            hip_path = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
            if hip_path and os.path.exists(hip_path):
                print_success("ROCm installation detected. Assuming AMD GPU present.")
                print_warning("Could not auto-detect GPU model. Please verify manually.")
                # Prompt user for GPU info
                if prompt_yes_no("Do you have an AMD Radeon RX 7900 XTX (24GB)?", default=True):
                    return True, "AMD Radeon RX 7900 XTX", 24.0
                else:
                    gpu_name = input("Enter your AMD GPU model (e.g., 'Radeon RX 7800 XT'): ").strip()
                    vram_input = input("Enter VRAM in GB (e.g., '16'): ").strip()
                    try:
                        vram_gb = float(vram_input)
                        return True, gpu_name, vram_gb
                    except ValueError:
                        return True, gpu_name, 0.0
    except Exception:
        pass

    print_step("No AMD GPU detected (rocm-smi not available)")
    return False, None, None


def check_wsl2_installed() -> bool:
    """Check if WSL2 is installed on Windows."""
    try:
        import platform
        if platform.system() != "Windows":
            return False

        output = run_command(["wsl", "--list", "--verbose"], capture=True, check=False)
        if output:
            # Windows WSL output sometimes has null bytes (appears as spaces between chars)
            # Normalize by removing null bytes and extra spaces
            normalized = output.replace('\x00', '').replace('  ', ' ')
            # Also try removing ALL spaces for pattern matching
            no_spaces = output.replace(' ', '').replace('\x00', '')

            # Check for VERSION header (handles both normal and spaced output)
            if "VERSION" in normalized or "VERSION" in no_spaces:
                # Check if any distro is using WSL2 (version 2)
                return "2" in output
        return False
    except Exception:
        return False


def install_wsl2_windows() -> bool:
    """Guide user through WSL2 installation on Windows."""
    print_header("WSL2 Installation Required")

    print("""
AMD GPU support on Windows requires WSL2 (Windows Subsystem for Linux 2).

This will:
  1. Install WSL2 with Ubuntu
  2. Require a system reboot
  3. After reboot, run this setup wizard again to install ROCm

""")

    if not prompt_yes_no("Install WSL2 now?", default=True):
        return False

    print_step("Installing WSL2...")
    print("This will open a PowerShell window with administrator privileges.")
    print()
    print_warning("IMPORTANT: WSL2 installation requires user interaction!")
    print("You will need to:")
    print("  1. Approve the UAC prompt")
    print("  2. Wait for WSL2 to download and install")
    print("  3. CREATE A UNIX USER ACCOUNT when prompted:")
    print("     - Enter a username (e.g., 'josh')")
    print("     - Enter a password")
    print("     - Confirm the password")
    print("  4. Wait for installation to complete")
    print("  5. Reboot your computer")
    print("  6. Run 'python setup.py' again after reboot")
    print()

    if not prompt_yes_no("Continue?", default=True):
        return False

    # Create a PowerShell script to install WSL2
    ps_script = """
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "  WSL2 Installation" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT: WSL2 will prompt you to create a Unix user account." -ForegroundColor Yellow
    Write-Host "You will need to:" -ForegroundColor Yellow
    Write-Host "  1. Enter a username (e.g., 'josh')" -ForegroundColor White
    Write-Host "  2. Enter a password" -ForegroundColor White
    Write-Host "  3. Confirm the password" -ForegroundColor White
    Write-Host ""
    Write-Host "This is normal and required. DO NOT close the window during setup." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Starting installation..." -ForegroundColor Green
    Write-Host ""

    try {
        wsl --install
        $exitCode = $LASTEXITCODE
        Write-Host ""
        Write-Host "======================================================================" -ForegroundColor Cyan

        if ($exitCode -eq 0) {
            Write-Host "Installation initiated successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "NEXT STEPS:" -ForegroundColor Yellow
            Write-Host "  1. Wait for installation to complete" -ForegroundColor White
            Write-Host "  2. REBOOT your computer" -ForegroundColor White
            Write-Host "  3. After reboot, run: python setup.py" -ForegroundColor White
        } else {
            Write-Host "Installation may have encountered issues (exit code: $exitCode)" -ForegroundColor Yellow
            Write-Host "Check the output above for details." -ForegroundColor White
        }
    } catch {
        Write-Host "ERROR: Installation failed" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to close this window"
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
        f.write(ps_script)
        ps_file = f.name

    try:
        # Run PowerShell as admin with -NoExit and -Wait to keep window open
        run_command([
            "powershell", "-Command",
            f"Start-Process powershell -ArgumentList '-NoExit -ExecutionPolicy Bypass -File {ps_file}' -Verb RunAs -Wait"
        ], check=False)

        print()
        print_warning("WSL2 installation started in PowerShell window.")
        print_warning("After installation completes and you reboot, run:")
        print_warning("  python setup.py")
        print()
        return False  # Return False to stop current setup
    except Exception as e:
        print_error(f"Failed to start WSL2 installation: {e}")
        print()
        print("Manual installation steps:")
        print("  1. Open PowerShell as Administrator")
        print("  2. Run: wsl --install")
        print("  3. Reboot your computer")
        print("  4. Run 'python setup.py' again")
        return False
    finally:
        try:
            os.unlink(ps_file)
        except:
            pass


def install_rocm_wsl2() -> bool:
    """Guide user through ROCm installation in WSL2."""
    print_header("ROCm Installation in WSL2")

    print("""
Installing ROCm in WSL2 will enable GPU acceleration for your AMD GPU.

This will:
  1. Install AMD GPU drivers in WSL2 Ubuntu
  2. Install ROCm 6.1+ toolkit
  3. Configure GPU access

This may take 10-20 minutes.
""")

    if not prompt_yes_no("Install ROCm in WSL2 now?", default=True):
        return False

    print_step("Installing ROCm in WSL2...")

    # ROCm installation script for WSL2
    wsl_commands = [
        "sudo apt-get update",
        "wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb",
        "sudo apt install -y ./amdgpu-install_6.1.60101-1_all.deb",
        "sudo amdgpu-install -y --usecase=rocm",
        "sudo usermod -a -G video $USER",
        "sudo usermod -a -G render $USER",
    ]

    try:
        for cmd in wsl_commands:
            print_step(f"Running: {cmd}")
            result = subprocess.run(
                ["wsl", "bash", "-c", cmd],
                capture_output=False,
                text=True
            )
            if result.returncode != 0:
                print_warning(f"Command failed (exit code {result.returncode}), continuing...")

        print_success("ROCm installation complete!")
        print()
        print_step("Verifying ROCm installation...")

        # Verify ROCm installation
        verify_result = run_command(
            ["wsl", "bash", "-c", "rocm-smi"],
            capture=True,
            check=False
        )

        if verify_result and "GPU" in verify_result:
            print_success("ROCm installed successfully!")
            print()
            print("Your AMD GPU should now be detected.")
            print("The setup wizard will continue with GPU configuration.")
            return True
        else:
            print_warning("ROCm installed but GPU not detected yet.")
            print_warning("You may need to:")
            print_warning("  1. Close and reopen your terminal")
            print_warning("  2. Restart Docker Desktop")
            print_warning("  3. Run 'python setup.py' again")
            return False

    except Exception as e:
        print_error(f"ROCm installation failed: {e}")
        print()
        print("Manual installation steps:")
        print("  1. Open WSL2: wsl")
        print("  2. Run these commands:")
        for cmd in wsl_commands:
            print(f"     {cmd}")
        return False


def detect_ram() -> float:
    """Detect total system RAM in GB."""
    print_step("Detecting system RAM...")
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    print_success(f"RAM detected: {ram_gb:.1f} GB")
    return ram_gb


def detect_cpu() -> int:
    """Detect CPU core count."""
    print_step("Detecting CPU cores...")
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    print_success(f"CPU cores detected: {cores}")
    return cores


def detect_hardware() -> Dict:
    """Detect all hardware."""
    print_header("Phase 2: Hardware Detection")

    # Try NVIDIA first, then AMD with ROCm
    has_nvidia, nvidia_name, nvidia_vram = detect_nvidia_gpu()
    has_amd_rocm, amd_rocm_name, amd_rocm_vram = detect_amd_gpu()

    # Also check for AMD GPU on Windows (even if ROCm not installed)
    import platform
    has_amd_windows, amd_windows_name, amd_windows_vram = False, None, None
    if platform.system() == "Windows" and not has_amd_rocm:
        has_amd_windows, amd_windows_name, amd_windows_vram = detect_amd_gpu_windows()

    # Determine which GPU to use
    has_gpu = has_nvidia or has_amd_rocm or has_amd_windows
    gpu_type = None
    gpu_name = None
    vram_gb = 0.0
    docker_runtime_ok = False
    use_native_ollama = False  # Use native Ollama instead of Docker (Windows with GPU)

    if has_nvidia:
        gpu_type = "nvidia"
        gpu_name = nvidia_name
        vram_gb = nvidia_vram or 0.0

        print_step("Checking nvidia-docker runtime...")
        try:
            output = run_command(["docker", "info", "--format", "{{.Runtimes}}"], capture=True)
            if "nvidia" in output:
                print_success("nvidia-docker runtime available")
                docker_runtime_ok = True
            else:
                print_warning("nvidia-docker runtime NOT available. Install nvidia-container-toolkit.")
                print_warning("GPU detection succeeded but Docker won't be able to use it without nvidia-docker.")
        except Exception:
            print_warning("Could not check Docker runtimes")

    elif has_amd_rocm:
        # AMD GPU with ROCm already installed
        gpu_type = "amd"
        gpu_name = amd_rocm_name
        vram_gb = amd_rocm_vram or 0.0

        print_success(f"AMD GPU with ROCm detected: {gpu_name}")
        hip_path = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
        if hip_path:
            print_success(f"ROCm installation found at: {hip_path}")
        docker_runtime_ok = True

    elif has_amd_windows:
        # AMD GPU on Windows - check if Ollama already has GPU access
        print_success(f"AMD GPU detected: {amd_windows_name}")
        if amd_windows_vram:
            print_success(f"VRAM detected: {amd_windows_vram:.1f} GB")

        print()
        print_step("Checking if native Ollama is using GPU...")

        ollama_using_gpu, processor_info = check_ollama_using_gpu()
        use_native_ollama = False  # Track if we should use native Ollama instead of Docker
        if ollama_using_gpu:
            print_success("Native Windows Ollama is already using your GPU!")
            print_success("No WSL2 or ROCm installation needed.")
            print()
            gpu_type = "amd"
            gpu_name = amd_windows_name
            vram_gb = amd_windows_vram or 0.0
            docker_runtime_ok = True
            use_native_ollama = True  # Use native Ollama, skip Docker Ollama
        else:
            # Ollama not using GPU - try WSL2 + ROCm approach
            print_warning("Ollama is not currently using GPU.")
            print_warning("Trying WSL2 + ROCm approach...")
            print()

            # Check if WSL2 is installed
            wsl2_installed = check_wsl2_installed()

            if not wsl2_installed:
                print_warning("WSL2 is not installed.")
                print()
                if prompt_yes_no("Would you like to install WSL2 and ROCm now?", default=True):
                    if install_wsl2_windows():
                        # WSL2 installation succeeded, continue
                        wsl2_installed = True
                    else:
                        # WSL2 installation requires reboot, exit setup
                        print()
                        print_warning("Setup will exit. Please reboot and run 'python setup.py' again.")
                        sys.exit(0)

            if wsl2_installed:
                # WSL2 installed, offer to install ROCm
                print_success("WSL2 is installed.")
                print()

                # Check if ROCm already installed in WSL2
                rocm_in_wsl2 = False
                try:
                    output = run_command(["wsl", "bash", "-c", "which rocm-smi"], capture=True, check=False)
                    rocm_in_wsl2 = output and "rocm-smi" in output
                except:
                    pass

                if not rocm_in_wsl2:
                    if prompt_yes_no("Would you like to install ROCm in WSL2 now?", default=True):
                        if install_rocm_wsl2():
                            # ROCm installed successfully
                            gpu_type = "amd"
                            gpu_name = amd_windows_name
                            vram_gb = amd_windows_vram or 0.0
                            docker_runtime_ok = True
                        else:
                            # ROCm installation failed or incomplete
                            print()
                            print_warning("Continuing with CPU-only mode.")
                            print_warning("You can install ROCm manually later and re-run setup.")
                    else:
                        print()
                        print_warning("Continuing with CPU-only mode.")
                        print_warning("To enable GPU later, install ROCm in WSL2 and re-run setup.")
                else:
                    # ROCm already installed
                    print_success("ROCm is already installed in WSL2!")
                    gpu_type = "amd"
                    gpu_name = amd_windows_name
                    vram_gb = amd_windows_vram or 0.0
                    docker_runtime_ok = True

    ram_gb = detect_ram()
    cpu_cores = detect_cpu()

    return {
        "has_gpu": has_gpu,
        "gpu_type": gpu_type,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "cpu_cores": cpu_cores,
        "docker_runtime_ok": docker_runtime_ok,
        "use_native_ollama": use_native_ollama,
    }


# ----------------------------
# Phase 3: Model Recommendations
# ----------------------------

def recommend_models(hw: Dict) -> Tuple[str, str, str, str, str]:
    """
    Recommend models based on hardware.
    Returns: (profile_name, extract_model, verify_model, write_model)
    """
    print_header("Phase 3: Model Recommendations")

    vram = hw["vram_gb"]
    ram = hw["ram_gb"]

    # Find best matching profile
    selected = None
    for min_vram, min_ram, profile, extract, verify, write in MODEL_RECOMMENDATION_MATRIX:
        if vram >= min_vram and ram >= min_ram:
            selected = (profile, extract, verify, write)
            break

    if not selected:
        # Fallback to minimum
        selected = MODEL_RECOMMENDATION_MATRIX[-1][2:]

    profile_name, extract_model, verify_model, write_model = selected

    print(f"Hardware profile: {profile_name}")
    print(f"  - GPU: {hw['gpu_name'] or 'None'}")
    print(f"  - VRAM: {vram:.1f} GB")
    print(f"  - RAM: {ram:.1f} GB")
    print(f"  - CPU cores: {hw['cpu_cores']}")
    print()
    print("Recommended models:")
    print(f"  - Extract: {extract_model}")
    print(f"  - Verify:  {verify_model}")
    print(f"  - Write:   {write_model}")
    print()

    if not prompt_yes_no("Use these models?", default=True):
        print("\nEnter custom models:")
        extract_model = input(f"  Extract model [{extract_model}]: ").strip() or extract_model
        verify_model = input(f"  Verify model  [{verify_model}]: ").strip() or verify_model
        write_model = input(f"  Write model   [{write_model}]: ").strip() or write_model

    print_success(f"Models selected: {extract_model}, {verify_model}, {write_model}")

    return profile_name, extract_model, verify_model, write_model


# ----------------------------
# Phase 4: Environment Configuration
# ----------------------------

def generate_env_file(hw: Dict, models: Tuple[str, str, str, str]) -> None:
    """Generate .env file with configuration."""
    print_header("Phase 4: Environment Configuration")

    profile_name, extract_model, verify_model, write_model = models

    env_content = f"""# Evident Video Fact Checker - Environment Configuration
# Generated by setup.py

# Service URLs (Docker internal network)
EVIDENT_OLLAMA_BASE_URL=http://ollama:11434
EVIDENT_SEARXNG_BASE_URL=http://searxng:8080

# Model selections
EVIDENT_MODEL_EXTRACT={extract_model}
EVIDENT_MODEL_VERIFY={verify_model}
EVIDENT_MODEL_WRITE={write_model}

# Model temperatures
EVIDENT_TEMPERATURE_EXTRACT=0.1
EVIDENT_TEMPERATURE_VERIFY=0.0
EVIDENT_TEMPERATURE_WRITE=0.5

# Hardware configuration
EVIDENT_GPU_ENABLED={'true' if hw['has_gpu'] and hw['docker_runtime_ok'] else 'false'}
EVIDENT_GPU_TYPE={hw.get('gpu_type') or 'none'}
EVIDENT_GPU_MEMORY_GB={hw['vram_gb']:.1f}
EVIDENT_RAM_GB={hw['ram_gb']:.1f}

# Pipeline budgets
EVIDENT_MAX_CLAIMS=25
EVIDENT_CACHE_TTL_DAYS=7

# Logging
EVIDENT_LOG_LEVEL=INFO
"""

    if os.path.exists(".env"):
        if not prompt_yes_no(".env file already exists. Overwrite?", default=False):
            print_warning("Skipping .env generation")
            return
        shutil.copy(".env", ".env.backup")
        print_step("Backed up existing .env to .env.backup")

    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)

    print_success(".env file created")


# ----------------------------
# Phase 5: Directory Structure
# ----------------------------

def create_directories() -> None:
    """Create required directory structure."""
    print_header("Phase 5: Directory Structure")

    dirs = [
        "data",
        "data/cache",
        "data/runs",
        "data/store",
        "data/inbox",
        "data/logs",
        "data/ollama",
        "searxng",
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print_step(f"Created: {dir_path}/")

    print_success("Directory structure created")


# ----------------------------
# Phase 6: Migration
# ----------------------------

def migrate_existing_data() -> None:
    """Migrate existing data from root to data/ subdirectory."""
    print_header("Phase 6: Data Migration")

    legacy_dirs = ["runs", "cache", "store", "inbox", "logs"]
    found_legacy = [d for d in legacy_dirs if os.path.exists(d) and os.path.isdir(d)]

    if not found_legacy:
        print_step("No existing data found. Skipping migration.")
        return

    print(f"Found existing data directories: {', '.join(found_legacy)}")
    if not prompt_yes_no("Migrate to data/ subdirectory?", default=True):
        print_warning("Skipping migration")
        return

    os.makedirs(".backup", exist_ok=True)

    for dir_name in found_legacy:
        src = dir_name
        dst = f"data/{dir_name}"
        backup = f".backup/{dir_name}"

        # Copy to data/
        if os.path.exists(dst):
            print_step(f"{dst} already exists, merging...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copytree(src, dst)
            print_step(f"Copied {src}/ → {dst}/")

        # Backup original
        if os.path.exists(backup):
            shutil.rmtree(backup)
        shutil.move(src, backup)
        print_step(f"Backed up {src}/ → {backup}/")

    print_success("Migration complete. Originals backed up to .backup/")


# ----------------------------
# Phase 7: SearXNG Configuration
# ----------------------------

def create_searxng_config() -> None:
    """Create SearXNG configuration files."""
    print_header("Phase 7: SearXNG Configuration")

    # Generate random secret key
    secret_key = secrets.token_hex(32)

    settings_yml = f"""# SearXNG settings for Evident Video Fact Checker
# Generated by setup.py

use_default_settings: true

server:
  secret_key: "{secret_key}"
  limiter: true
  image_proxy: true

redis:
  url: redis://redis:6379/0

ui:
  static_use_hash: true

search:
  safe_search: 0
  autocomplete: ""

enabled_plugins:
  - 'Hash plugin'
  - 'Self Information'
  - 'Tracker URL remover'

engines:
  - name: google
    disabled: false
  - name: bing
    disabled: false
  - name: duckduckgo
    disabled: false
  - name: wikipedia
    disabled: false
"""

    limiter_toml = """# SearXNG rate limiter configuration

[botdetection.ip_limit]
link_token = true
"""

    with open("searxng/settings.yml", "w", encoding="utf-8") as f:
        f.write(settings_yml)
    print_success("Created searxng/settings.yml")

    with open("searxng/limiter.toml", "w", encoding="utf-8") as f:
        f.write(limiter_toml)
    print_success("Created searxng/limiter.toml")


# ----------------------------
# Phase 8: Service Initialization
# ----------------------------

def start_services(hw: Dict) -> bool:
    """Start Docker services."""
    print_header("Phase 8: Service Initialization")

    use_native_ollama = hw.get("use_native_ollama", False)

    # Determine compose files
    compose_files = ["-f", "docker-compose.yml"]
    if use_native_ollama:
        # Windows with native Ollama using GPU - use native-ollama compose
        compose_files.extend(["-f", "docker-compose.native-ollama.yml"])
        print_step("Using native Ollama mode (GPU already working)")
        print_success("Native Ollama at localhost:11434 will be used")
        print()
    elif hw["has_gpu"] and hw["docker_runtime_ok"]:
        gpu_type = hw.get("gpu_type", "nvidia")
        if gpu_type == "nvidia":
            compose_files.extend(["-f", "docker-compose.gpu.yml"])
            print_step("Using NVIDIA GPU configuration")
        elif gpu_type == "amd":
            compose_files.extend(["-f", "docker-compose.amd.yml"])
            print_step("Using AMD GPU configuration with ROCm")
            print_warning("AMD GPU support requires:")
            print_warning("  - ROCm 6.1+ installed on host")
            print_warning("  - WSL2 with ROCm (Windows) or native ROCm (Linux)")
            print_warning("  - Ollama with ROCm support")
            print()
            if not prompt_yes_no("Continue with AMD GPU setup?", default=True):
                print_step("Falling back to CPU-only mode")
                compose_files = ["-f", "docker-compose.yml"]
    else:
        print_step("Using CPU-only configuration")

    # Start Redis
    print_step("Starting Redis...")
    try:
        run_command(["docker", "compose"] + compose_files + ["up", "-d", "redis"])
        print_success("Redis started")
    except subprocess.CalledProcessError:
        print_error("Failed to start Redis")
        return False

    if use_native_ollama:
        # Check that native Ollama is running
        print_step("Checking native Ollama...")
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                print_success("Native Ollama is running")
        except Exception as e:
            print_error("Native Ollama is not running!")
            print_error("Please start Ollama and run setup again.")
            print_error("  - On Windows: Start Ollama from the system tray or run 'ollama serve'")
            return False
    else:
        # Start Docker Ollama
        print_step("Starting Ollama...")
        try:
            run_command(["docker", "compose"] + compose_files + ["up", "-d", "ollama"])
            print_success("Ollama started")
        except subprocess.CalledProcessError:
            print_error("Failed to start Ollama")
            return False

        # Wait for Ollama healthcheck
        print_step("Waiting for Ollama to be ready...")
        import time
        max_wait = 120
        waited = 0
        while waited < max_wait:
            try:
                output = run_command(
                    ["docker", "compose"] + compose_files + ["ps", "--format", "json"],
                    capture=True
                )
                services = json.loads(f"[{output.replace('}{', '},{')}]")
                ollama_svc = next((s for s in services if s.get("Service") == "ollama"), None)
                if ollama_svc and ollama_svc.get("Health") == "healthy":
                    print_success("Ollama is ready")
                    break
            except Exception:
                pass
            time.sleep(5)
            waited += 5
            print(f"  Waiting... ({waited}s/{max_wait}s)")
        else:
            print_warning("Ollama did not become healthy within timeout. Continuing anyway.")

    return True


# ----------------------------
# Phase 9: Model Download
# ----------------------------

def download_models(models: Tuple[str, str, str, str], hw: Dict) -> bool:
    """Download Ollama models."""
    print_header("Phase 9: Model Download")

    _, extract_model, verify_model, write_model = models
    unique_models = list(set([extract_model, verify_model, write_model]))

    # Check which models are already available
    use_native_ollama = hw.get("use_native_ollama", False)
    existing_models = set()

    # Determine compose files for Docker mode
    compose_files = ["-f", "docker-compose.yml"]
    if not use_native_ollama and hw["has_gpu"] and hw["docker_runtime_ok"]:
        gpu_type = hw.get("gpu_type", "nvidia")
        if gpu_type == "nvidia":
            compose_files.extend(["-f", "docker-compose.gpu.yml"])
        elif gpu_type == "amd":
            compose_files.extend(["-f", "docker-compose.amd.yml"])

    if use_native_ollama:
        # Check native Ollama for existing models
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for m in data.get("models", []):
                    existing_models.add(m.get("name", ""))
        except Exception:
            pass

    # Filter to models that need downloading
    needed_models = [m for m in unique_models if m not in existing_models]
    already_have = [m for m in unique_models if m in existing_models]

    if already_have:
        print(f"Already have: {', '.join(already_have)}")

    if not needed_models:
        print_success("All models already available!")
        return True

    print(f"Downloading {len(needed_models)} model(s): {', '.join(needed_models)}")
    print("This may take 30-120 minutes depending on your connection.\n")

    for model in needed_models:
        print_step(f"Pulling {model}...")
        try:
            if use_native_ollama:
                # Use native ollama CLI
                subprocess.run(["ollama", "pull", model], check=True)
            else:
                # Use Docker Ollama
                subprocess.run(
                    ["docker", "compose"] + compose_files + ["exec", "-T", "ollama", "ollama", "pull", model],
                    check=True
                )
            print_success(f"{model} downloaded")
        except subprocess.CalledProcessError:
            print_error(f"Failed to download {model}")
            return False

    print_success("All models downloaded")
    return True


# ----------------------------
# Phase 10: SearXNG Startup & Validation
# ----------------------------

def start_searxng_and_validate(hw: Dict) -> bool:
    """Start SearXNG and validate all services."""
    print_header("Phase 10: SearXNG Startup & Validation")

    compose_files = ["-f", "docker-compose.yml"]
    if hw["has_gpu"] and hw["docker_runtime_ok"]:
        gpu_type = hw.get("gpu_type", "nvidia")
        if gpu_type == "nvidia":
            compose_files.extend(["-f", "docker-compose.gpu.yml"])
        elif gpu_type == "amd":
            compose_files.extend(["-f", "docker-compose.amd.yml"])

    # Start SearXNG
    print_step("Starting SearXNG...")
    try:
        run_command(["docker", "compose"] + compose_files + ["up", "-d", "searxng"])
        print_success("SearXNG started")
    except subprocess.CalledProcessError:
        print_error("Failed to start SearXNG")
        return False

    # Wait for healthcheck
    print_step("Waiting for SearXNG to be ready...")
    import time
    max_wait = 60
    waited = 0
    while waited < max_wait:
        try:
            output = run_command(
                ["docker", "compose"] + compose_files + ["ps", "--format", "json"],
                capture=True
            )
            services = json.loads(f"[{output.replace('}{', '},{')}]")
            searxng_svc = next((s for s in services if s.get("Service") == "searxng"), None)
            if searxng_svc and searxng_svc.get("Health") == "healthy":
                print_success("SearXNG is ready")
                break
        except Exception:
            pass
        time.sleep(5)
        waited += 5
        print(f"  Waiting... ({waited}s/{max_wait}s)")
    else:
        print_warning("SearXNG did not become healthy within timeout")

    # Validate services
    print_step("Validating services...")

    # Test Ollama
    try:
        output = run_command(
            ["docker", "compose"] + compose_files + ["exec", "-T", "ollama", "curl", "-f", "http://localhost:11434/api/tags"],
            capture=True
        )
        if output:
            print_success("Ollama API responding")
    except subprocess.CalledProcessError:
        print_error("Ollama API not responding")
        return False

    # Test SearXNG
    try:
        output = run_command(
            ["docker", "compose"] + compose_files + ["exec", "-T", "searxng", "curl", "-f", "http://localhost:8080/search?q=test&format=json"],
            capture=True
        )
        if output:
            print_success("SearXNG API responding")
    except subprocess.CalledProcessError:
        print_warning("SearXNG API not responding (may need more time)")

    return True


# ----------------------------
# Main
# ----------------------------

def main():
    """Run interactive setup wizard."""
    print("""
=======================================================================

              Evident Video Fact Checker - Interactive Setup Wizard

  This wizard will:
    1. Check prerequisites (Docker, Docker Compose, Python)
    2. Detect your hardware (GPU, RAM, CPU)
    3. Recommend models based on your system
    4. Configure environment variables
    5. Create directory structure
    6. Migrate existing data (if any)
    7. Start Docker services (Redis, Ollama, SearXNG)
    8. Download Ollama models (30-120 minutes)
    9. Validate setup

=======================================================================
""")

    if not prompt_yes_no("Continue with setup?", default=True):
        print("Setup cancelled.")
        sys.exit(0)

    # Phase 1: Prerequisites
    prereqs = check_prerequisites()
    # Hard requirements: Docker, Docker Compose, Python
    # Soft requirements: FFmpeg (only needed for YouTube Whisper fallback)
    hard_reqs = {k: v for k, v in prereqs.items() if k != "ffmpeg"}
    if not all(hard_reqs.values()):
        print_error("Missing prerequisites. Please install required tools and try again.")
        sys.exit(1)
    if not prereqs.get("ffmpeg"):
        print_warning("Setup will continue without FFmpeg. Install it later for YouTube Whisper support.")

    # Phase 2: Hardware
    hw = detect_hardware()

    # Phase 3: Model recommendations
    models = recommend_models(hw)

    # Phase 4: Environment configuration
    generate_env_file(hw, models)

    # Phase 5: Directory structure
    create_directories()

    # Phase 6: Migration
    migrate_existing_data()

    # Phase 7: SearXNG config
    create_searxng_config()

    # Phase 8: Service initialization
    if not start_services(hw):
        print_error("Service initialization failed")
        sys.exit(1)

    # Phase 9: Model download
    if not download_models(models, hw):
        print_error("Model download failed")
        sys.exit(1)

    # Phase 10: SearXNG & validation
    if not start_searxng_and_validate(hw):
        print_warning("Validation had warnings, but setup may still work")

    # Success
    print_header("Setup Complete!")
    print("""
=======================================================================
                           SETUP COMPLETE!
=======================================================================

Next steps:

1. Add a transcript to data/inbox/:
   cp /path/to/transcript.txt data/inbox/

2. Run the pipeline:
   make run ARGS="--infile data/inbox/transcript.txt --channel YourChannel"

   Or with review mode:
   make review ARGS="--infile data/inbox/transcript.txt"

3. Check outputs:
   ls data/runs/

Useful commands:
  make status   - Show service status
  make logs     - View all logs
  make stop     - Stop all services
  make start    - Start all services
  make models   - List downloaded models
  make help     - Show all commands

For more info, see:
  - DOCKER.md     - Docker architecture and operations
  - MIGRATION.md  - Migration guide for existing users
  - README.md     - General project documentation
""")

    print_success("Setup wizard finished successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
