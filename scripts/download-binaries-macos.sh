#!/bin/sh

# This script must be run from the scripts directory.

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script is only for macOS."
    exit 1
fi

if [ "$(uname -m)" != "arm64" ]; then
    echo "This script requires an Apple Silicon Mac (arm64)."
    echo "Detected architecture: $(uname -m)"
    exit 1
fi

# Check if curl exists
if command -v curl >/dev/null 2>&1; then
    echo "curl is installed."
else
    echo "curl is not installed. Please install it."
    exit 1
fi

# Check if wget exists
if command -v wget >/dev/null 2>&1; then
    echo "wget is installed."
else
    echo "wget is not installed. Please install it."
    exit 1
fi

ARCHIVE="py2d-macos-arm64-binaries.tar.gz"
EXTRACTED_DIR="py2d-macos-arm64-binaries"

# Remove existing binaries
if [ -d rcssserver ]; then
    echo "rcssserver directory exists, remove it"
    rm -rf rcssserver
fi

if [ -d proxy ]; then
    echo "proxy directory exists, remove it"
    rm -rf proxy
fi

# Download macOS ARM64 binaries

wget "https://github.com/TheAbry/COL333-Assignment-3-py2d-fork/releases/latest/download/$ARCHIVE"

# Check download was successful
if [ ! -f "$ARCHIVE" ]; then
    echo "Download failed"
    exit 1
fi

# Extract archive
echo "Extracting binaries..."
tar -xvf "$ARCHIVE"

# Check extracted directory exists
if [ ! -d "rcssserver" ]; then
    echo "Extraction failed: rcssserver not found."
    exit 1
fi

if [ ! -d "proxy" ]; then
    echo "Extraction failed: proxy not found."
    exit 1
fi

rm -f "$ARCHIVE"

chmod +x rcssserver/rcssserver
chmod +x rcssserver/squashfs-root/AppRun
chmod +x proxy/sample_player
chmod +x proxy/sample_coach
chmod +x proxy/sample_trainer

# Copy custom formations
cp -r ../my-formations/* ./proxy

echo "macOS ARM64 binaries installed successfully."