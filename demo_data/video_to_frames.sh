#!/usr/bin/env bash
set -x

FPS=30
CURRENT_DIR=$(dirname $0)
OUTPUT_DIR=${CURRENT_DIR}/video_frames

for VIDEO in ${CURRENT_DIR}/videos/*.mp4; do
    FILENAME=$(basename ${VIDEO})
    VIDEO_ID="${FILENAME%%_*}"
    mkdir -p ${OUTPUT_DIR}/${VIDEO_ID}
    ffmpeg -y -i ${VIDEO} -r ${FPS} -q:v 1 -start_number 0 ${OUTPUT_DIR}/${VIDEO_ID}/%05d.jpg
done