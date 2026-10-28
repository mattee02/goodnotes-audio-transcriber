#!/usr/bin/env bash
# ==============================================================================
# GoodNotes Audio Processor & Transcriber - Wrapper Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Automatically include Rust/Cargo in PATH if installed in user's home directory
if [ -d "$HOME/.cargo/bin" ] && [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 2. Check minimal system prerequisites
if ! command -v ffmpeg &>/dev/null; then
    echo "[ERROR] 'ffmpeg' is not installed or not in PATH." >&2
    echo "Please install FFmpeg using your system package manager (e.g. sudo dnf install ffmpeg / sudo apt install ffmpeg)" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] 'python3' is not installed." >&2
    exit 1
fi

# 3. Virtual Environment Setup (.venv)
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Initializing Python Virtual Environment (.venv) ==="
    # Prefer Python 3.11 if present: PyPI distributes precompiled binary wheels for DeepFilterNet
    # on Python <= 3.11, avoiding the need for a local Rust compiler.
    if command -v python3.11 &>/dev/null; then
        echo "Found python3.11: using it to create virtualenv for optimal precompiled wheel compatibility..."
        python3.11 -m venv "$VENV_DIR"
    else
        echo "Creating virtualenv with $(python3 --version)..."
        python3 -m venv "$VENV_DIR"
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 4. Verify dependencies via the main module
if ! python3 -c "import process_notes" &>/dev/null; then
    echo "=== Installing missing Python dependencies into .venv ==="
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MINOR" -ge 12 ] && ! command -v cargo &>/dev/null; then
        echo "----------------------------------------------------------------------"
        echo "[NOTICE] On Python >= 3.12 DeepFilterNet may require Rust (cargo) to build deepfilterlib."
        echo "If installation fails, you can either:"
        echo "  - Install Rust toolchain: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  - Or install python3.11 (which uses prebuilt wheels): sudo dnf install python3.11"
        echo "----------------------------------------------------------------------"
    fi

    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# 5. Execute the pipeline forwarding all CLI arguments
exec python3 "$SCRIPT_DIR/process_notes.py" "$@"
