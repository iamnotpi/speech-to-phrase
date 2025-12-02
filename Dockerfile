FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml setup.cfg ./
COPY speech_to_phrase/ ./speech_to_phrase/

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Default command (will be overridden by add-on config)
CMD ["python", "-m", "speech_to_phrase", \
     "--uri", "tcp://0.0.0.0:10300", \
     "--train-dir", "/data/train", \
     "--tools-dir", "/data/tools", \
     "--models-dir", "/data/models", \
     "--hass-websocket-uri", "ws://homeassistant.local:8123/api/websocket", \
     "--hass-token", "${HASS_TOKEN}", \
     "--retrain-on-start"]