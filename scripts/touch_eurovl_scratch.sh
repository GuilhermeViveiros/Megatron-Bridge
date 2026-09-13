#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Resets mtimes on every file that still lives physically on $SCRATCH, to keep
# it out of the filesystem's 30/90-day auto-purge window.
#
# `EuroVL-Data/raw-data` and `EuroVL-Data/processed-data` are symlinks to
# /e/data1/smurf4eu-data (a persistent, project-lifetime, non-age-purged
# fileset) -- plain `find` does not follow symlinks into directories, so
# those ~63 TiB are skipped automatically. Only real directories still on
# scratch get touched.
#
# usage: touch_eurovl_scratch.sh [ROOT ...]
#   ROOT defaults to both EuroVL-Data and viveiros1 under the project scratch.
#
# Prefer running this via run_touch_eurovl_scratch.sh, which backgrounds it
# with nohup so it survives terminal/session teardown.

set -euo pipefail

if [ "$#" -gt 0 ]; then
    ROOTS=("$@")
else
    ROOTS=(
        /e/scratch/e-ext-2025e01-100/EuroVL-Data
        /e/scratch/e-ext-2025e01-100/viveiros1
    )
fi

for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || { echo "error: root directory not found: $root" >&2; exit 1; }
done

echo "counting files under: ${ROOTS[*]} (symlinked dirs skipped)"
total=0
for root in "${ROOTS[@]}"; do
    n=$(find "$root" -type f | wc -l)
    echo "  $root: ${n} files"
    total=$((total + n))
done
echo "total: ${total} files"

# tqdm-style progress bar: [#####-----] 42% (12345/29000) 812 files/s ETA 00:12:03
draw_progress() {
    local done_n=$1 total_n=$2 start_ts=$3
    local now elapsed rate eta pct filled width bar
    now=$(date +%s)
    elapsed=$(( now - start_ts ))
    [ "$elapsed" -lt 1 ] && elapsed=1
    rate=$(( done_n / elapsed ))
    if [ "$total_n" -gt 0 ]; then
        pct=$(( done_n * 100 / total_n ))
    else
        pct=100
    fi
    if [ "$rate" -gt 0 ] && [ "$total_n" -gt "$done_n" ]; then
        eta=$(( (total_n - done_n) / rate ))
    else
        eta=0
    fi
    width=30
    filled=$(( pct * width / 100 ))
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar="${bar}$(printf '%*s' $((width - filled)) '')"
    printf '\r[%s] %3d%% (%d/%d) %d files/s ETA %02d:%02d:%02d' \
        "$bar" "$pct" "$done_n" "$total_n" "$rate" \
        $((eta / 3600)) $((eta % 3600 / 60)) $((eta % 60))
}

count=0
start=$(date +%s)
last_draw=0

for root in "${ROOTS[@]}"; do
    while IFS= read -r -d '' f; do
        touch -- "$f"
        count=$((count + 1))
        now=$(date +%s)
        if [ "$now" -gt "$last_draw" ]; then
            draw_progress "$count" "$total" "$start"
            last_draw=$now
        fi
    done < <(find "$root" -type f -print0)
done

draw_progress "$count" "$total" "$start"
echo ""

elapsed=$(( $(date +%s) - start ))
echo "done: touched ${count} files under ${ROOTS[*]} in ${elapsed}s"
