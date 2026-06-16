#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"

if [ -z "$IMAGE_NAME" ]; then
  echo "用法：IMAGE_NAME=dockerhub用户名/镜像名 [IMAGE_TAG=版本] ./publish-dockerhub.sh"
  echo "示例：IMAGE_NAME=yourname/moss-tts-nano IMAGE_TAG=cpu-$(date +%Y%m%d) ./publish-dockerhub.sh"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未找到 docker 命令。"
  exit 1
fi

echo "构建并推送镜像：${IMAGE_NAME}:${IMAGE_TAG}"
echo "平台：${PLATFORM}"

docker buildx create --use --name moss-tts-builder >/dev/null 2>&1 || docker buildx use moss-tts-builder
docker buildx build \
  --platform "$PLATFORM" \
  -f Dockerfile.moss-tts-cpu \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  --push \
  .

echo "推送完成：${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "部署时可使用："
echo "  MOSS_TTS_IMAGE=${IMAGE_NAME}:${IMAGE_TAG} docker compose -f docker-compose.image.yml up -d"
