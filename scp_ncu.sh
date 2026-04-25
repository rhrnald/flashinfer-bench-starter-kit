#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-smartcho:/home/snu_avq1/workspace/chaewon/flashinfer-bench-starter-kit/ex.ncu-rep}"
DST="${2:-./$(date +%y%m%d_%H%M).ncu-rep}"

scp "${SRC}" "${DST}"
echo "Copied ${SRC} -> ${DST}"
