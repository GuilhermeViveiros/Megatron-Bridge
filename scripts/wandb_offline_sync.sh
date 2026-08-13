#!/usr/bin/env bash
# Periodically upload offline W&B runs to the cloud from an internet-connected (login) node.
#
# Compute nodes have no internet, so training logs W&B *offline* (WANDB_MODE=offline in
# apptainer.sh) into <run>/wandb/offline-run-*. This loop, run on a login node that DOES have
# internet, `wandb sync`s those run dirs on an interval so the W&B UI stays roughly live while
# the job trains.
#
# IMPORTANT: re-syncing an already-fully-synced run dir is NOT a no-op in practice — it was
# observed to leave repeated/duplicate entries in the W&B UI for the same run. So this loop
# tracks, per run dir, the mtime of its run-*.wandb file at last sync (under
# $ROOT/.wandb_sync_state/) and only calls `wandb sync` again when that file has grown since —
# i.e. only while the run is still actively being written. Finished runs get synced once and
# then left alone.
#
# Usage (from a login node):
#   nohup scripts/wandb_offline_sync.sh > "$SCRATCH/euro_vl_runs/wandb_sync.log" 2>&1 &
#   # stop it later with:  kill <pid>
#
# Requires the wandb CLI on PATH (login node):  pip install wandb
#
# Env overrides:
#   WANDB_SYNC_ROOT   dir tree to scan for offline runs (default: $SCRATCH/euro_vl_runs)
#   WANDB_SYNC_EVERY  seconds between passes (default: 300)
set -uo pipefail

# Absolute default (do NOT derive from $SCRATCH — on login nodes it is the bare project scratch
# without the user subdir; this matches apptainer.sh's hardcoded SCRATCH). Override with
# WANDB_SYNC_ROOT if your runs live elsewhere.
ROOT="${WANDB_SYNC_ROOT:-/e/scratch/e-ext-2025e01-100/viveiros1/euro_vl_runs}"
EVERY="${WANDB_SYNC_EVERY:-300}"

# Load WANDB_API_KEY from the gitignored .env at the repo root (needed to upload).
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set (put it in $REPO/.env)" >&2
    exit 1
fi
# Resolve wandb to an ABSOLUTE path once (the venv lives on shared /e, reachable from any login
# node). Relying on PATH is fragile across JUPITER's multiple login nodes / fresh sessions — that
# is what silently broke the loop before ("wandb: command not found" mid-run). Override with
# WANDB_BIN if your install lives elsewhere.
WANDB_BIN="${WANDB_BIN:-/e/project1/e-ext-2025e01-100/viveiros1/envs/eurovlm/bin/wandb}"
if [ ! -x "$WANDB_BIN" ]; then
    WANDB_BIN="$(command -v wandb 2>/dev/null || true)"
fi
if [ -z "$WANDB_BIN" ] || [ ! -x "$WANDB_BIN" ]; then
    echo "ERROR: wandb CLI not found (set WANDB_BIN, or pip install wandb)" >&2
    exit 1
fi

export WANDB_MODE=online  # wandb sync uploads regardless, but be explicit
echo "[wandb-sync] root=$ROOT every=${EVERY}s wandb=$WANDB_BIN pid=$$"
trap 'echo "[wandb-sync] stopping (pid $$)"; exit 0' INT TERM

STATE_DIR="$ROOT/.wandb_sync_state"
mkdir -p "$STATE_DIR"

while true; do
    ts="$(date '+%F %T')"
    # Offline run dirs are named offline-run-* (or run-* once started); each holds a .wandb file.
    mapfile -t runs < <(find "$ROOT" -type d \( -name 'offline-run-*' -o -name 'run-*' \) 2>/dev/null | sort)
    if [ "${#runs[@]}" -eq 0 ]; then
        echo "[$ts] no offline runs under $ROOT yet"
    else
        # Only (re)sync a run dir when its .wandb file has grown since the last sync — an
        # unchanged file means nothing new to upload, and re-syncing it anyway is what produced
        # duplicate entries in the W&B UI.
        to_sync=()
        for run in "${runs[@]}"; do
            wandb_file="$(find "$run" -maxdepth 1 -name '*.wandb' -print -quit 2>/dev/null)"
            [ -z "$wandb_file" ] && continue
            mtime="$(stat -c %Y "$wandb_file" 2>/dev/null || stat -f %m "$wandb_file" 2>/dev/null)"
            state_file="$STATE_DIR/$(basename "$run").mtime"
            last_mtime="$(cat "$state_file" 2>/dev/null || echo '')"
            if [ "$mtime" != "$last_mtime" ]; then
                to_sync+=("$run")
            fi
        done
        if [ "${#to_sync[@]}" -eq 0 ]; then
            echo "[$ts] ${#runs[@]} run(s) known, nothing new to sync"
        else
            echo "[$ts] syncing ${#to_sync[@]}/${#runs[@]} run(s) with new data"
            # Sync one run at a time so each can be retried/recovered independently.
            for run in "${to_sync[@]}"; do
                name="$(basename "$run")"
                altid_file="$STATE_DIR/$name.altid"
                sync_args=("$run")
                if [ -f "$altid_file" ]; then
                    # A previous pass hit "previously created and deleted" for this run's
                    # original id and minted a replacement id (below) — keep reusing that same
                    # replacement id on every later pass so we don't mint a new one (and a new
                    # duplicate cloud run) each time.
                    sync_args=(--id "$(cat "$altid_file")" "$run")
                fi
                out="$("$WANDB_BIN" sync "${sync_args[@]}" 2>&1)"
                echo "$out" | sed 's/^/[wandb] /'
                if echo "$out" | grep -q 'previously created and deleted' && [ ! -f "$altid_file" ]; then
                    # The run's original id was deleted from the W&B UI (e.g. earlier duplicate
                    # cleanup) but this run dir is still live/growing locally — retrying under the
                    # same id will never succeed. Mint a stable replacement id, persist it, and
                    # retry immediately so this pass doesn't silently drop the run's data.
                    new_id="${name##*-}-resynced"
                    echo "$new_id" > "$altid_file"
                    echo "[wandb] original run id deleted upstream; retrying as $new_id" | sed 's/^/[wandb] /'
                    out="$("$WANDB_BIN" sync --id "$new_id" "$run" 2>&1)"
                    echo "$out" | sed 's/^/[wandb] /'
                fi
                if ! echo "$out" | grep -qi 'error'; then
                    wandb_file="$(find "$run" -maxdepth 1 -name '*.wandb' -print -quit 2>/dev/null)"
                    [ -z "$wandb_file" ] && continue
                    mtime="$(stat -c %Y "$wandb_file" 2>/dev/null || stat -f %m "$wandb_file" 2>/dev/null)"
                    echo "$mtime" > "$STATE_DIR/$name.mtime"
                fi
                # On failure, deliberately do NOT write the .mtime state file — leave the run
                # eligible for retry next pass rather than silently abandoning its data.
            done
        fi
    fi
    sleep "$EVERY"
done
