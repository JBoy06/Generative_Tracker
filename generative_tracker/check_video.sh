#!/bin/bash
# Quick video diagnostics script

if [ -z "$1" ]; then
    echo "Usage: $0 video_file.mp4"
    exit 1
fi

VIDEO="$1"

if [ ! -f "$VIDEO" ]; then
    echo "Error: Video file '$VIDEO' not found"
    exit 1
fi

echo "=== Video Diagnostics for: $VIDEO ==="
echo

# Basic file info
echo "File size: $(du -h "$VIDEO" | cut -f1)"
echo

# Video stream info
echo "=== Video Stream Info ==="
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration,nb_frames,r_frame_rate -of default=noprint_wrappers=1 "$VIDEO"
echo

# Audio stream info
echo "=== Audio Stream Info ==="
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,duration,sample_rate -of default=noprint_wrappers=1 "$VIDEO" 2>/dev/null || echo "No audio stream found"
echo

# Container info
echo "=== Container Info ==="
ffprobe -v error -show_entries format=format_name,duration,size,bit_rate -of default=noprint_wrappers=1 "$VIDEO"
echo

# Frame count verification
FRAME_COUNT=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$VIDEO" 2>/dev/null)
echo "Actual frame count: $FRAME_COUNT"

# Check if video is playable by extracting first frame
echo "Testing video playability..."
ffmpeg -i "$VIDEO" -vframes 1 -f image2 -y /tmp/test_frame.jpg >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Video is readable and can extract frames"
    rm -f /tmp/test_frame.jpg
else
    echo "✗ Video appears to be corrupted or unreadable"
fi

echo
echo "=== Summary ==="
if [ "$FRAME_COUNT" -gt 0 ]; then
    echo "✓ Video contains $FRAME_COUNT frames"
    echo "✓ File appears to be valid"
    echo
    echo "If the video appears empty in your player, try:"
    echo "1. Playing with VLC media player"
    echo "2. Using: ffplay \"$VIDEO\""
    echo "3. Converting to standard format: ffmpeg -i \"$VIDEO\" -c:v libx264 -c:a aac \"${VIDEO%.mp4}_converted.mp4\""
else
    echo "✗ Video contains no frames"
    echo "✗ This explains why the video appears empty"
fi