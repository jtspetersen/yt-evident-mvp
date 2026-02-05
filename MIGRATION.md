# Migration Guide - Evident Video Fact Checker Docker

This guide helps existing Evident Video Fact Checker users migrate from native Python setup to the Docker Compose stack.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Backup Your Data](#backup-your-data)
- [Migration Steps](#migration-steps)
- [Command Mapping](#command-mapping)
- [File Path Changes](#file-path-changes)
- [Configuration Changes](#configuration-changes)
- [Verification](#verification)
- [Rollback](#rollback)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

1. **Docker** (20.10+)
   - Download: https://docs.docker.com/get-docker/
   - Verify: `docker --version`

2. **Docker Compose** (v2.0+)
   - Included with Docker Desktop
   - Verify: `docker compose version` (note: no hyphen)

3. **Python 3.11+** (for setup wizard only)
   - Verify: `python --version`
   - Install psutil: `pip install psutil`

### Optional (for GPU acceleration)

4. **nvidia-docker** (NVIDIA Container Toolkit)
   - Only needed if you have an NVIDIA GPU
   - Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   - Verify: `docker info | grep -i runtime` (should show `nvidia`)

---

## Backup Your Data

**IMPORTANT:** Before migrating, back up your existing data.

```bash
# Create backup directory
mkdir -p .backup

# Backup runtime data
cp -r runs .backup/
cp -r cache .backup/
cp -r store .backup/
cp -r inbox .backup/
cp -r logs .backup/ 2>/dev/null || true

# Backup configuration
cp config.yaml .backup/
cp .venv/Scripts/activate .backup/ 2>/dev/null || true  # Windows
cp .venv/bin/activate .backup/ 2>/dev/null || true      # Linux/Mac
```

Your backups are now in `.backup/`. Keep this directory until you've verified the Docker setup works.

---

## Migration Steps

### Step 1: Pull Latest Code

```bash
git pull origin master
```

### Step 2: Run Setup Wizard

The interactive wizard will:
- Detect your hardware (GPU, RAM, CPU)
- Recommend appropriate models
- Migrate your existing data to `data/` subdirectory
- Configure environment variables
- Start services
- Download models

Run the wizard:

```bash
python setup.py
```

**What to expect:**

1. **Prerequisites Check** - Verifies Docker, Docker Compose, Python
2. **Hardware Detection** - Scans for GPU, RAM, CPU
3. **Model Recommendations** - Suggests models based on your hardware
   - Press `Y` to accept or `N` to customize
4. **Environment Configuration** - Creates `.env` file
   - If `.env` exists, you'll be asked to overwrite
5. **Directory Structure** - Creates `data/` and `searxng/` directories
6. **Data Migration** - Moves `runs/`, `cache/`, `store/` to `data/` subdirectories
   - Original directories backed up to `.backup/`
7. **Service Initialization** - Starts Redis and Ollama
8. **Model Download** - Downloads selected models (30-120 minutes)
9. **SearXNG Initialization** - Starts SearXNG service
10. **Validation** - Tests all services

### Step 3: Verify Setup

Check that all services are healthy:

```bash
make status
```

Expected output:
```
NAME                IMAGE                    STATUS         HEALTH
evident-redis       redis:7-alpine           Up 2 minutes   healthy
evident-ollama      ollama/ollama:latest     Up 2 minutes   healthy
evident-searxng     searxng/searxng:latest   Up 1 minute    healthy
```

### Step 4: Test Run

Add a test transcript to `data/inbox/`:

```bash
cp .backup/inbox/some-transcript.txt data/inbox/
```

Run the pipeline:

```bash
make run ARGS="--infile data/inbox/some-transcript.txt --channel TestChannel"
```

Check outputs:

```bash
ls data/runs/
```

---

## Command Mapping

| **Old (Native)** | **New (Docker)** | **Notes** |
|------------------|------------------|-----------|
| `./run.sh` | `make run ARGS="..."` | Must specify ARGS |
| `./runvid` | `make run ARGS="--infile data/inbox/file.txt"` | Path prefix: `data/` |
| `python -m app.main --infile inbox/file.txt` | `make run ARGS="--infile data/inbox/file.txt"` | Add `data/` prefix |
| `python -m app.main --review` | `make review ARGS="--infile data/inbox/file.txt"` | Separate review command |
| `git pull && ./run.sh` | `git pull && make build && make run ARGS="..."` | Rebuild after code changes |
| *N/A* | `make setup` | First-time setup wizard |
| *N/A* | `make start` | Start services |
| *N/A* | `make stop` | Stop services |
| *N/A* | `make logs` | View logs |
| *N/A* | `make models` | List models |
| *N/A* | `make status` | Check health |

### Common Operations

**Start services:**
```bash
make start
```

**Stop services:**
```bash
make stop
```

**View logs:**
```bash
make logs              # All services
make logs-app          # App only
make logs-ollama       # Ollama only
make logs-searxng      # SearXNG only
```

**Run pipeline:**
```bash
make run ARGS="--infile data/inbox/transcript.txt --channel YourChannel"
```

**Interactive review mode:**
```bash
make review ARGS="--infile data/inbox/transcript.txt"
```

**List models:**
```bash
make models
```

**Download additional model:**
```bash
make models-pull MODEL=qwen3:14b
```

**Rebuild after code changes:**
```bash
make build
make start
```

---

## File Path Changes

### Directory Structure

**Before:**
```
evident-video-fact-checker/
├── inbox/          # Input transcripts
├── runs/           # Output artifacts
├── cache/          # URL cache
├── store/          # JSONL logs
├── logs/           # Run logs
└── app/            # Source code
```

**After:**
```
evident-video-fact-checker/
├── data/                   # All runtime data (NEW)
│   ├── inbox/             # Input transcripts
│   ├── runs/              # Output artifacts
│   ├── cache/             # URL cache
│   ├── store/             # JSONL logs
│   ├── logs/              # Run logs
│   └── ollama/            # Ollama models
├── searxng/               # SearXNG config (NEW)
├── docker-compose.yml     # Base services (NEW)
├── Dockerfile             # App container (NEW)
├── Makefile               # Operations (NEW)
├── .env                   # Environment vars (NEW)
└── app/                   # Source code
```

### Path Updates

| **Before** | **After** | **Reason** |
|------------|-----------|------------|
| `inbox/file.txt` | `data/inbox/file.txt` | Centralized data directory |
| `runs/20240101_120000__channel__video/` | `data/runs/20240101_120000__channel__video/` | Centralized data directory |
| `cache/` | `data/cache/` | Centralized data directory |
| `store/` | `data/store/` | Centralized data directory |
| `logs/` | `data/logs/` | Centralized data directory |

**Note:** The setup wizard automatically migrates your existing data to the new structure.

---

## Configuration Changes

### Before (Native Python)

**Configuration source:** `config.yaml` only

**Service URLs:**
- Ollama: `http://localhost:11434`
- SearXNG: `http://localhost:8080`

**Running services:**
- Manually start Ollama: `ollama serve`
- Manually start SearXNG: (varies by installation)

### After (Docker Compose)

**Configuration hierarchy (highest to lowest precedence):**
1. Environment variables (in shell)
2. `.env` file (gitignored)
3. `config.yaml` (lowest priority, backward compatible)

**Service URLs:**
- Inside Docker network:
  - Ollama: `http://ollama:11434`
  - SearXNG: `http://searxng:8080`
- From host machine (for testing):
  - Ollama: `http://localhost:11434`
  - SearXNG: `http://localhost:8080`

**Running services:**
- Automatic via Docker Compose: `make start`
- Services start on boot if Docker Desktop is running

### Environment Variables

**.env file example:**
```bash
# Service URLs
EVIDENT_OLLAMA_BASE_URL=http://ollama:11434
EVIDENT_SEARXNG_BASE_URL=http://searxng:8080

# Models
EVIDENT_MODEL_EXTRACT=qwen3:8b
EVIDENT_MODEL_VERIFY=qwen3:30b
EVIDENT_MODEL_WRITE=gemma3:27b

# Temperatures
EVIDENT_TEMPERATURE_EXTRACT=0.1
EVIDENT_TEMPERATURE_VERIFY=0.0
EVIDENT_TEMPERATURE_WRITE=0.5

# Hardware
EVIDENT_GPU_ENABLED=true
EVIDENT_GPU_MEMORY_GB=24.0
EVIDENT_RAM_GB=64.0

# Budgets
EVIDENT_MAX_CLAIMS=25
EVIDENT_CACHE_TTL_DAYS=7

# Logging
EVIDENT_LOG_LEVEL=INFO
```

**Note:** `.env` is created automatically by `setup.py`. You can edit it manually to override models or settings.

---

## Verification

### Check Services

```bash
make status
```

All services should show `healthy` status.

### Test Ollama

```bash
docker compose exec ollama ollama list
```

Should show your downloaded models.

### Test SearXNG

```bash
curl "http://localhost:8080/search?q=test&format=json"
```

Should return JSON with search results.

### Run Pipeline

```bash
# Add a test file
echo "This is a test transcript." > data/inbox/test.txt

# Run pipeline
make run ARGS="--infile data/inbox/test.txt --channel Test"

# Check output
ls data/runs/
```

Should create a timestamped directory with artifacts.

---

## Rollback

If you need to revert to native Python setup:

### Step 1: Stop Docker Services

```bash
make stop
```

### Step 2: Restore Backups

```bash
# Restore data directories
cp -r .backup/runs ./
cp -r .backup/cache ./
cp -r .backup/store ./
cp -r .backup/inbox ./
cp -r .backup/logs ./ 2>/dev/null || true

# Restore config
cp .backup/config.yaml ./
```

### Step 3: Restore Virtual Environment

```bash
# Windows
.venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate

# Reinstall dependencies if needed
pip install -r Requirements.txt
```

### Step 4: Restart Native Services

```bash
# Start Ollama (native)
ollama serve

# Start SearXNG (varies by installation)
# ...

# Run pipeline (native)
python -m app.main --infile inbox/transcript.txt
```

### Step 5: Clean Up Docker (Optional)

```bash
# Remove all Docker data
make clean-all

# Remove Docker files
rm -rf data/
rm -rf searxng/
rm docker-compose.yml docker-compose.gpu.yml Dockerfile .dockerignore .env Makefile setup.py
```

---

## Troubleshooting

### Issue: Services won't start

**Symptoms:**
- `make start` fails
- `docker compose up` errors

**Solutions:**

1. Check Docker is running:
   ```bash
   docker info
   ```

2. Check port conflicts:
   ```bash
   # Ollama port 11434
   netstat -an | grep 11434

   # SearXNG port 8080
   netstat -an | grep 8080
   ```

3. Stop native services if running:
   ```bash
   # Stop native Ollama
   pkill ollama

   # Stop native SearXNG
   # (varies by installation)
   ```

4. Restart Docker Desktop

### Issue: Ollama not responding

**Symptoms:**
- Pipeline hangs during extract/verify/write stages
- `curl http://localhost:11434/api/tags` times out

**Solutions:**

1. Check Ollama logs:
   ```bash
   make logs-ollama
   ```

2. Restart Ollama:
   ```bash
   docker compose restart ollama
   ```

3. Check healthcheck:
   ```bash
   docker compose ps
   ```
   Ollama should show `healthy` status after 30-60 seconds.

4. Test manually:
   ```bash
   docker compose exec ollama ollama list
   ```

### Issue: Models not downloading

**Symptoms:**
- `setup.py` hangs during model download
- `make models-pull` fails

**Solutions:**

1. Check network connectivity:
   ```bash
   docker compose exec ollama curl https://ollama.com
   ```

2. Try downloading manually:
   ```bash
   docker compose exec ollama ollama pull qwen3:8b
   ```

3. Check disk space:
   ```bash
   df -h
   # Models can be 5-20 GB each
   ```

4. Increase timeout and retry:
   - Models download from Ollama's CDN
   - Large models (30B) can take 30-60 minutes on slow connections

### Issue: SearXNG not responding

**Symptoms:**
- Pipeline hangs during evidence retrieval
- `curl http://localhost:8080/search?q=test` fails

**Solutions:**

1. Check SearXNG logs:
   ```bash
   make logs-searxng
   ```

2. Restart SearXNG:
   ```bash
   docker compose restart searxng
   ```

3. Check Redis:
   ```bash
   docker compose ps redis
   ```
   Redis must be healthy for SearXNG to work.

4. Verify configuration:
   ```bash
   cat searxng/settings.yml
   ```
   Ensure `secret_key` is set and `redis.url` points to `redis://redis:6379/0`.

### Issue: "No such file or directory" errors

**Symptoms:**
- `make run` fails with "file not found"
- Pipeline can't find `config.yaml`

**Solutions:**

1. Ensure you're in project root:
   ```bash
   pwd
   # Should show: /path/to/evident-video-fact-checker
   ```

2. Check file paths use `data/` prefix:
   ```bash
   # Wrong
   make run ARGS="--infile inbox/file.txt"

   # Correct
   make run ARGS="--infile data/inbox/file.txt"
   ```

3. Check config.yaml exists in project root:
   ```bash
   ls -la config.yaml
   ```

### Issue: GPU not detected

**Symptoms:**
- Setup wizard shows "No GPU detected"
- GPU works on host but not in Docker

**Solutions for NVIDIA GPU:**

1. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Verify Docker runtime:
   ```bash
   docker info | grep -i runtime
   # Should show: nvidia
   ```

3. Test GPU in container:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. Restart setup:
   ```bash
   python setup.py
   ```

**Solutions for AMD GPU (ROCm):**

1. Install ROCm 6.1+:
   ```bash
   # Ubuntu 22.04
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
   sudo apt install ./amdgpu-install_6.1.60101-1_all.deb
   sudo amdgpu-install --usecase=rocm
   ```

   **Windows:** Install ROCm in WSL2:
   ```bash
   # In WSL2 Ubuntu
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.1.60101-1_all.deb
   sudo apt install ./amdgpu-install_6.1.60101-1_all.deb
   sudo amdgpu-install --usecase=rocm
   ```

2. Verify ROCm installation:
   ```bash
   rocm-smi
   # Should show your AMD GPU
   ```

3. Add user to video group:
   ```bash
   sudo usermod -a -G video $USER
   sudo usermod -a -G render $USER
   # Log out and back in
   ```

4. Test GPU:
   ```bash
   rocm-smi --showproductname
   ```

5. Restart setup:
   ```bash
   python setup.py
   ```

**Note for AMD RX 7900 XTX users:**
- Your GPU is supported with ROCm 6.1+
- GFX version: 11.0.0 (gfx1100)
- Setup wizard will auto-detect if ROCm is installed
- Standard Ollama image may not include ROCm support
- May need to modify docker-compose.amd.yml to use ROCm-enabled image

### Issue: Permission denied errors

**Symptoms:**
- `make run` fails with "permission denied"
- Can't write to `data/runs/`

**Solutions:**

1. Check directory ownership:
   ```bash
   ls -la data/
   ```

2. Fix permissions:
   ```bash
   # Linux/Mac
   sudo chown -R $USER:$USER data/

   # Windows (Git Bash/WSL)
   # Usually not an issue
   ```

3. Check Docker volume mounts in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./data/runs:/app/runs  # Ensure this is not read-only
   ```

### Getting Help

If you encounter issues not covered here:

1. Check Docker logs:
   ```bash
   make logs
   ```

2. Check service health:
   ```bash
   make status
   ```

3. Verify environment:
   ```bash
   cat .env
   ```

4. Open an issue:
   - GitHub: https://github.com/your-org/evident-video-fact-checker/issues
   - Include:
     - Error messages
     - Output of `make status`
     - Output of `docker compose logs`
     - Your hardware (GPU/RAM/CPU)
     - OS and Docker version

---

## Summary

**Key Changes:**

✓ **Setup:** One command (`python setup.py`) instead of manual service configuration
✓ **Services:** Managed by Docker Compose (`make start/stop`) instead of manual processes
✓ **Paths:** Add `data/` prefix to all file paths
✓ **Commands:** Use `make` commands instead of `./run.sh` or direct Python
✓ **Config:** Edit `.env` file instead of `config.yaml`
✓ **Models:** Managed via `make models` commands

**Benefits:**

✓ Consistent environment across systems
✓ No Python virtual environment management
✓ Automatic service startup
✓ Easy model management
✓ GPU support auto-detected
✓ One-command setup for new systems

**Need Help?**

- Read [DOCKER.md](DOCKER.md) for Docker architecture details
- Run `make help` for command reference
- Check logs with `make logs`
- Open an issue on GitHub

---

**Migration complete! Enjoy your containerized Evident Video Fact Checker.**
