#!/bin/sh
set -e

# Expect a long-lived Home Assistant token in HASS_TOKEN
if [ -z "$HASS_TOKEN" ]; then
  echo "HASS_TOKEN is empty. Provide a long-lived HA token via env." >&2
  exit 1
fi

# Allow overriding the HA websocket URI via HASS_WS_URI
: "${HASS_WS_URI:=ws://homeassistant:8123/api/websocket}"

exec python -m speech_to_phrase \
  --uri tcp://0.0.0.0:10300 \
  --train-dir /data/train \
  --tools-dir /data/tools \
  --models-dir /data/models \
  --hass-websocket-uri "$HASS_WS_URI" \
  --hass-token "$HASS_TOKEN"