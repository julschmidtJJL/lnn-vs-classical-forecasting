# Reproducible cloud build for the Bachelor thesis experiment (CPU-only).
# Target: linux/amd64 (Hetzner CPX31 etc.). On Apple Silicon, build with
#   docker build --platform linux/amd64 ...
# so the image matches the x86_64 VM it will run on.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the CPU build of torch first (the default PyPI wheel is the multi-GB
# CUDA build we don't need). requirements.txt then sees torch==2.2.2 satisfied.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# Relative imports + relative config path require running from src/.
WORKDIR /app/src

# Secrets are NOT baked in (.env is dockerignored). Pass them at run time:
#   docker run --rm --env-file .env <image>                      # smoke (default)
#   docker run --rm --env-file .env <image> python main.py config.yaml   # full grid
CMD ["python", "main.py", "config.smoke.yaml"]
