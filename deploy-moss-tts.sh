#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.moss-tts.yml}"
CONTAINER_NAME="${MOSS_TTS_CONTAINER:-moss-tts-nano-cpu}"
APP_PORT="${MOSS_TTS_PORT:-8000}"
APP_URL="http://127.0.0.1:${APP_PORT}/"

if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未找到 docker 命令。请先安装 Docker Desktop 并确保当前 shell 可以访问 docker。"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "错误：docker compose 不可用。请确认 Docker Desktop 已启动。"
  exit 1
fi

echo "停止并清理旧容器..."
docker compose -f "$COMPOSE_FILE" down

echo "开始构建 MOSS-TTS-Nano Docker 镜像 (CPU版本，本地源码)..."
docker compose -f "$COMPOSE_FILE" build --pull

echo "启动容器..."
docker compose -f "$COMPOSE_FILE" up -d

echo "查看容器状态..."
docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "等待服务响应 /health ..."
for i in $(seq 1 40); do
  if docker exec "$CONTAINER_NAME" curl -fsS http://127.0.0.1:18083/health >/dev/null 2>&1; then
    echo "服务已响应。"
    break
  fi
  if [ "$i" -eq 40 ]; then
    echo "服务暂未响应，可能仍在下载模型或预热。请继续查看日志。"
  fi
  sleep 3
done

echo ""
echo "最近日志："
docker logs --tail 80 "$CONTAINER_NAME" || true

echo ""
echo "部署完成！"
echo ""
echo "访问地址："
echo "  $APP_URL"
echo ""
echo "常用命令："
echo "  进入容器: docker exec -it $CONTAINER_NAME bash"
echo "  查看日志: docker compose -f $COMPOSE_FILE logs -f"
echo "  停止服务: docker compose -f $COMPOSE_FILE down"
echo "  重启服务: docker compose -f $COMPOSE_FILE restart"
