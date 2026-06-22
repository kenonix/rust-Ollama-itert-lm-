#!/bin/bash
set -e

# Configuration
VERSION="v0.13.1"
CACHE_DIR="$HOME/.cache/litert-lm"
SRC_DIR="$CACHE_DIR/src/LiteRT-LM"
LIB_DIR="$CACHE_DIR/lib"
INCLUDE_DIR="$CACHE_DIR/include"
BAZELISK_VERSION="v1.20.0"

echo "=== LiteRT-LM C++ Library Build Script ==="
echo "Targeting version: $VERSION"

# 1. Check/Install System Dependencies
echo "[1/5] Checking system dependencies..."
MISSING_DEPS=()
for cmd in git python3 clang clang++; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING_DEPS+=("$cmd")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo "Error: Missing required system dependencies: ${MISSING_DEPS[*]}"
    echo "Please install them first. For example, on Ubuntu/Debian:"
    echo "  sudo apt-get update && sudo apt-get install -y git python3 clang lld"
    exit 1
fi
echo "System dependencies OK."

# 2. Check/Download Bazelisk
echo "[2/5] Setting up Bazelisk..."
if command -v bazelisk &>/dev/null; then
    BAZEL_CMD="bazelisk"
elif command -v bazel &>/dev/null; then
    BAZEL_CMD="bazel"
else
    # Detect Arch
    ARCH=$(uname -m)
    BAZELISK_BIN="bazelisk-linux-amd64"
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        BAZELISK_BIN="bazelisk-linux-arm64"
    fi
    
    BAZELISK_PATH="$CACHE_DIR/bin/bazelisk"
    mkdir -p "$(dirname "$BAZELISK_PATH")"
    if [ ! -f "$BAZELISK_PATH" ]; then
        echo "Downloading bazelisk ($BAZELISK_BIN) to $BAZELISK_PATH..."
        curl -L -o "$BAZELISK_PATH" "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/$BAZELISK_BIN"
        chmod +x "$BAZELISK_PATH"
    fi
    BAZEL_CMD="$BAZELISK_PATH"
fi
echo "Using Bazel command: $BAZEL_CMD"

# 3. Clone/Checkout LiteRT-LM
echo "[3/5] Checking LiteRT-LM source code..."
mkdir -p "$CACHE_DIR/src"
if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning LiteRT-LM repository..."
    git clone https://github.com/google-ai-edge/LiteRT-LM.git "$SRC_DIR"
fi

cd "$SRC_DIR"
echo "Fetching tags..."
git fetch --tags
echo "Checking out version $VERSION..."
git checkout "$VERSION" || git checkout main

# 4. Build C shared library
echo "[4/5] Building LiteRT-LM C Shared Library via Bazel..."
echo "This might take a while (typically 15-45 minutes depending on CPU)..."

# Note: We build using opt mode. --define=litert_link_capi_so=true resolves GPU symbols.
# On Linux, OpenCL is disabled by default in .bazelrc, but if we want GPU support on ARM64,
# LiteRT will dynamically load OpenCL or Vulkan drivers on the target system.
$BAZEL_CMD build //c:litert-lm -c opt

# 5. Copy Build Artifacts
echo "[5/5] Deploying build artifacts..."
mkdir -p "$LIB_DIR"
mkdir -p "$INCLUDE_DIR"

SO_FILE="bazel-bin/c/liblitert-lm.so"
if [ ! -f "$SO_FILE" ]; then
    # Fallback/MacOS check
    if [ -f "bazel-bin/c/liblitert-lm.dylib" ]; then
        SO_FILE="bazel-bin/c/liblitert-lm.dylib"
    elif [ -f "bazel-bin/c/litert-lm.dll" ]; then
        SO_FILE="bazel-bin/c/litert-lm.dll"
    fi
fi

if [ -f "$SO_FILE" ]; then
    cp -v "$SO_FILE" "$LIB_DIR/"
    cp -v "c/engine.h" "$INCLUDE_DIR/"
    echo "Success! Shared library is copied to: $LIB_DIR/$(basename "$SO_FILE")"
    echo "Header file is copied to: $INCLUDE_DIR/engine.h"
    echo ""
    echo "To run the Rust server with this library, add this to your environment:"
    echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$LIB_DIR"
else
    echo "Error: Shared library build finished but output file was not found!"
    exit 1
fi
