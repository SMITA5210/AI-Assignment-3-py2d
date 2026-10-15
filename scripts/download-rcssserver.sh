#!/bin/sh

# check rcssserver directory exists, if exists, remove it
if [ -d rcssserver ]; then
    echo "rcssserver directory exists, remove it"
    rm -rf rcssserver
fi

mkdir rcssserver

cd rcssserver

# Check if curl exists
if command -v curl >/dev/null 2>&1; then
    echo "curl is installed."
else
    echo "curl is not installed. Please install it."
    exit 1
fi

# Check if get exists
if command -v wget >/dev/null 2>&1; then
    echo "wget is installed."
else
    echo "wget is not installed. Please install it."
    exit 1
fi

# download soccer simulation server App Image
# pinned to the version the autograder uses
wget https://github.com/CLSFramework/rcssserver/releases/download/19.0.0/rcssserver-x86_64-19.0.0.AppImage

# check download is successful
if [ ! -f *.AppImage ]; then
    echo "Download failed"
    exit 1
fi

mv rcssserver-x86_64-*.AppImage rcssserver

chmod +x rcssserver