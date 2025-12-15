# Use HA add-on base (set BUILD_FROM per arch if needed)
ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:latest
FROM ${BUILD_FROM}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.cfg ./
COPY speech_to_phrase/ ./speech_to_phrase/
COPY custom_intents/data/vi.json /usr/local/lib/python3.12/site-packages/home_assistant_intents/data/vi.json

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir home-assistant-intents==2025.6.23 && \
    pip install --no-cache-dir -e .

# Copy run script and set as entrypoint
COPY rootfs/run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]