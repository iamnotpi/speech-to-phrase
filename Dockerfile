# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml setup.cfg ./
COPY speech_to_phrase/ ./speech_to_phrase/

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the package
RUN pip install --no-cache-dir -e .

# Install home-assistant-intents
RUN pip install --no-cache-dir home-assistant-intents==2025.6.23

# Copy your custom intents file (override the default)
COPY custom_intents/data/vi.json /usr/local/lib/python3.12/site-packages/home_assistant_intents/data/vi.json

# Stage 2: Runtime
FROM python:3.12-slim

# Only install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Default command (fixed hostname: homeassistant instead of homeassistant.local)
CMD ["python", "-m", "speech_to_phrase", \
     "--uri", "tcp://0.0.0.0:10300", \
     "--train-dir", "/data/train", \
     "--tools-dir", "/data/tools", \
     "--models-dir", "/data/models", \
     "--hass-websocket-uri", "ws://homeassistant:8123/api/websocket", \
     "--hass-token", "${HASS_TOKEN}", \
     "--retrain-on-start"]