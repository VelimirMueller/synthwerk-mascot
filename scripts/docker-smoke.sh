#!/usr/bin/env bash
# Build the mascot image, start it with NO network (`--network none`) and a read-only root
# file system, then run scripts/smoke_check.py inside it: health, search quality, no egress,
# rate limit. The build stage of ci.yml runs it; it also runs locally with Docker.
#
# Usage: scripts/docker-smoke.sh   (exit 0 = every check passed)
set -euo pipefail
cd "$(dirname "$0")/.."

commit=${GITHUB_SHA:-$(git rev-parse --verify -q HEAD 2>/dev/null || echo 0000000)}
image=synthwerk-mascot:smoke
docker build --build-arg COMMIT="$commit" -t "$image" .
docker image inspect "$image" --format 'image size: {{.Size}} bytes'

name="mascot-smoke-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$name" --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --memory 1g \
  -e SYNTHWERK_MASCOT_RATE_LIMIT_PER_CLIENT=20 \
  -v "$PWD/scripts:/smoke:ro" "$image" >/dev/null

network=$(docker inspect "$name" --format '{{.HostConfig.NetworkMode}}')
[ "$network" = "none" ] || { echo "::error::container network is $network, want none"; exit 1; }
user=$(docker exec "$name" id -u)
[ "$user" != "0" ] || { echo "::error::container runs as root"; exit 1; }

if ! docker exec "$name" python /smoke/smoke_check.py; then
  docker logs "$name" | tail -n 50
  echo "::error::smoke checks failed"
  exit 1
fi
docker stats --no-stream --format 'memory: {{.MemUsage}}' "$name"
