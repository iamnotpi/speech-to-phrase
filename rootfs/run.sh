#!/usr/bin/with-contenv bashio

TOKEN="$(bashio::config 'hass_token')"
if [ -z "$TOKEN" ]; then
  bashio::exit.nok "hass_token is empty. Please set it in the add-on config."
fi

exec python -m speech_to_phrase \
  --uri tcp://0.0.0.0:10300 \
  --train-dir /data/train \
  --tools-dir /data/tools \
  --models-dir /data/models \
  --hass-websocket-uri ws://homeassistant:8123/api/websocket \
  --hass-token "$TOKEN" \
  --retrain-on-start

