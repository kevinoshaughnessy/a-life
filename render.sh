#!/bin/sh
# Render the film, then compress it to about 28 MB with a two-pass encode.
set -e
cd "$(dirname "$0")"
python3 life.py video build/a_life_master.mp4
cd build
ffmpeg -v error -y -i a_life_master.mp4 -c:v libx264 -preset slow -b:v 1800k -pass 1 -tune animation -pix_fmt yuv420p -an -f null /dev/null
ffmpeg -v error -y -i a_life_master.mp4 -c:v libx264 -preset slow -b:v 1800k -pass 2 -tune animation -pix_fmt yuv420p -c:a copy -movflags +faststart a_life_1080p.mp4
rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree
echo "build/a_life_1080p.mp4"
