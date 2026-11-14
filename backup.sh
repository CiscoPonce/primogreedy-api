#!/bin/bash
# PrimoGreedy — Daily DuckDB Backup
set -euo pipefail

ROOT="/home/ubuntu/primogreedy"
BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M)
BACKUP_FILE="${BACKUP_DIR}/primogreedy_data_${DATE}.duckdb"

mkdir -p "$BACKUP_DIR"
echo "[$(date -Iseconds)] Starting PrimoGreedy backup..."

docker stop primogreedy-api >/dev/null
trap "docker start primogreedy-api >/dev/null" EXIT

cp "${ROOT}/data.duckdb" "$BACKUP_FILE"
docker start primogreedy-api >/dev/null
trap - EXIT

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[$(date -Iseconds)] Backup complete: ${BACKUP_FILE} (${SIZE})"

DELETED=$(find "$BACKUP_DIR" -name "primogreedy_data_*.duckdb" -mtime +14 -print -delete | wc -l)
echo "[$(date -Iseconds)] Cleaned up ${DELETED} old backups"
