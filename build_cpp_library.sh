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
echo "Pulling Git LFS objects..."
git lfs install || true
git lfs pull

# 4. Build C shared library
echo "[4/5] Building LiteRT-LM C Shared Library via Bazel..."
echo "This might take a while (typically 15-45 minutes depending on CPU)..."

# The shared-library target is defined in python/litert_lm/BUILD.
$BAZEL_CMD build //python/litert_lm:litert-lm -c opt

# 5. Copy Build Artifacts
echo "[5/5] Deploying build artifacts..."
mkdir -p "$LIB_DIR"
mkdir -p "$INCLUDE_DIR"

SO_FILE=""
for candidate in \
    "bazel-bin/python/litert_lm/liblitert-lm.so" \
    "bazel-bin/python/litert_lm/litert-lm.so" \
    "bazel-bin/python/litert_lm/litert-lm"; do
    if [ -f "$candidate" ]; then
        SO_FILE="$candidate"
        break
    fi
done

if [ -z "$SO_FILE" ]; then
    # Fallback/MacOS check
    for candidate in \
        "bazel-bin/python/litert_lm/liblitert-lm.dylib" \
        "bazel-bin/python/litert_lm/litert-lm.dylib" \
        "bazel-bin/python/litert_lm/litert-lm.dll"; do
        if [ -f "$candidate" ]; then
            SO_FILE="$candidate"
            break
        fi
    done
fi

if [ -f "$SO_FILE" ]; then
    cp -vf "$SO_FILE" "$LIB_DIR/"
    cp -vf "c/engine.h" "$INCLUDE_DIR/"
    
    # Also copy libGemmaModelConstraintProvider.so
    PREBUILT_ARCH="linux_x86_64"
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        PREBUILT_ARCH="linux_arm64"
    fi
    GEMMA_SO="prebuilt/${PREBUILT_ARCH}/libGemmaModelConstraintProvider.so"
    if [ -f "$GEMMA_SO" ]; then
        cp -vf "$GEMMA_SO" "$LIB_DIR/"
        echo "Copied $GEMMA_SO to $LIB_DIR/"
    fi

    echo "Success! Shared library is copied to: $LIB_DIR/$(basename "$SO_FILE")"
    echo "Header file is copied to: $INCLUDE_DIR/engine.h"
    echo ""
    echo "To run the Rust server with this library, add this to your environment:"
    echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$LIB_DIR"
else
    echo "Error: Shared library build finished but output file was not found!"
    exit 1
fi
