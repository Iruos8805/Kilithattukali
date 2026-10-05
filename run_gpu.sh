#!/usr/bin/env bash
# Source or execute commands with AMD ROCm GPU acceleration enabled for Navi 33 / gfx1102 / RDNA3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH="${SCRIPT_DIR}/.rocm-libs/opt/rocm-7.0.0/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ $# -gt 0 ]; then
    exec "$@"
fi
