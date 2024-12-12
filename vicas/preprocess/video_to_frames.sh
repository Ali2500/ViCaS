#!/usr/bin/env bash
set -eux

# Provide the following two path variables
# Directory containing video files
VIDEOS_DIR=$1
# Directory where video frames will be written
OUTPUT_FRAMES_DIR=$2

FPS=30

for VIDEO in ${VIDEOS_DIR}/*.mp4; do
    FILENAME=$(basename ${VIDEO})
    VIDEO_ID="${FILENAME%%_*}"
    mkdir -p ${OUTPUT_FRAMES_DIR}/${VIDEO_ID}
    ffmpeg -y -i ${VIDEO} -r ${FPS} -q:v 1 -start_number 0 ${OUTPUT_FRAMES_DIR}/${VIDEO_ID}/%05d.jpg
done