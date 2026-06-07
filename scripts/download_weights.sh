#!/bin/bash
# 下载 Real-ESRGAN 超分权重

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/../sr/weights"

mkdir -p "${WEIGHTS_DIR}"

URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
TARGET="${WEIGHTS_DIR}/realesr-general-x4v3.pth"

if [ -f "${TARGET}" ]; then
    echo "权重已存在: ${TARGET}"
    exit 0
fi

echo "下载 Real-ESRGAN 权重..."
wget -q --show-progress "${URL}" -O "${TARGET}"

if [ $? -eq 0 ]; then
    echo "下载完成: ${TARGET}"
    echo "大小: $(du -h ${TARGET} | cut -f1)"
else
    echo "下载失败"
    exit 1
fi
