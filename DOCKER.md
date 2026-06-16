# Docker usage

## Build and run from this source tree

```bash
./deploy-moss-tts.sh
```

Open:

```text
http://127.0.0.1:8000/
```

The local compose file builds the image from this repository, not from the upstream OpenMOSS GitHub repository.

## Push a prebuilt image to Docker Hub

Login first:

```bash
docker login
```

Build and push:

```bash
IMAGE_NAME=yourname/moss-tts-nano IMAGE_TAG=cpu-latest ./publish-dockerhub.sh
```

## Run from a pushed image without rebuilding

```bash
MOSS_TTS_IMAGE=yourname/moss-tts-nano:cpu-latest docker compose -f docker-compose.image.yml up -d
```

Useful overrides:

```bash
MOSS_TTS_PORT=8001
HF_ENDPOINT=https://hf-mirror.com
```

The app listens on `0.0.0.0:18083` inside the container so Docker port mapping can expose it to the host.

Custom voices created from the web UI are persisted in `./moss-tts-data/custom_voices.json` and `./moss-tts-data/custom_audio` when using either compose file.
