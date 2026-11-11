#!/usr/bin/env bash
# 把仓库里的 deploy/tensorrt/artifacts/{onnx,engines,calib,...}
# 链到公用目录，避免每人复制 51GB。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SHARED="${OPENPI_TRT_SHARED:-/share_data/public/models/openpi/pi05_libero_tensorrt}"
LOCAL="$ROOT/artifacts"

if [[ ! -d "$SHARED" ]]; then
  echo "找不到公用目录: $SHARED"
  exit 1
fi

mkdir -p "$LOCAL"
for name in onnx engines calib engines_legacy_attn_fp8 logs \
            action_compare.json segment_rmse.json; do
  src="$SHARED/$name"
  dst="$LOCAL/$name"
  if [[ ! -e "$src" ]]; then
    echo "跳过（公用目录没有）: $name"
    continue
  fi
  if [[ -L "$dst" ]]; then
    echo "已是软链接: $dst"
    continue
  fi
  if [[ -e "$dst" ]]; then
    echo "本地已有实体文件，不覆盖: $dst"
    echo "  若确认用公用份:  rm -rf $dst && $0"
    continue
  fi
  ln -s "$src" "$dst"
  echo "链接 $dst -> $src"
done
echo "完成。推理脚本仍使用 deploy/tensorrt/artifacts ，无需改代码。"
