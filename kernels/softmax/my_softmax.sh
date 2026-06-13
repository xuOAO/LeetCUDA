#!/usr/bin/env bash
set -e

name=${1:-safe_softmax_f16x8_pack_f32_per_token}
dtype=${2:-float16}

ncu --nvtx \
  --nvtx-include "profiling/" \
  --set full \
  --import-source yes \
  -o "$name" \
  -- python3 my_softmax.py --profiling "$name" --dtype "$dtype"
