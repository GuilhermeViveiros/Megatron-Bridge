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

# Launches touch_eurovl_scratch.sh in the background so it survives
# terminal/session teardown. Plain `nohup ... & disown` is not enough for
# long-running jobs on this cluster (killed jobs seen before) -- use `setsid`
# so the process gets its own session, detached from the controlling TTY.
#
# usage: run_touch_eurovl_scratch.sh [ROOT ...]
#   Forwards ROOT args to touch_eurovl_scratch.sh; defaults to EuroVL-Data
#   and viveiros1 under the project scratch if none given.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOUCH_SCRIPT="$SCRIPT_DIR/touch_eurovl_scratch.sh"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/touch_eurovl_scratch_$(date +%Y%m%d_%H%M%S).log"

setsid nohup "$TOUCH_SCRIPT" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
disown

echo "started touch_eurovl_scratch.sh in background"
echo "  pid: $pid"
echo "  log: $LOG_FILE"
echo ""
echo "follow progress with: tail -f $LOG_FILE"
echo "check it's alive with: ps -p $pid"
