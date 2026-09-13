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

"""Pick checkpoints at ~25/50/75/100% of a training run's *available* checkpoint
history (not of some assumed absolute iteration count — early checkpoints are
routinely pruned, so "25% of final iter" can point at a directory that no
longer exists).

Example:
  uv run python -m evals.scripts.select_checkpoints /scratch/euro_vl_runs/qwen3_pa_pyav
"""

import argparse
import re
from pathlib import Path


PERCENTILES = (10, 50, 100)


def list_checkpoints(run_dir: str) -> list[Path]:
    """Return iter_* checkpoint dirs under ``run_dir``, sorted by iteration number."""
    dirs = [p for p in Path(run_dir).glob("iter_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(re.search(r"iter_(\d+)", p.name).group(1)))


def select(run_dir: str) -> dict[int, Path]:
    """Map each percentile in ``PERCENTILES`` to a checkpoint path, by list index."""
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise ValueError(f"No iter_* checkpoints found under {run_dir}")
    n = len(checkpoints)
    selected = {}
    for pct in PERCENTILES:
        idx = min(n - 1, round((pct / 100) * (n - 1)))
        selected[pct] = checkpoints[idx]
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Directory containing iter_* checkpoint subdirs.")
    args = parser.parse_args()
    for pct, path in select(args.run_dir).items():
        print(f"{pct}\t{path}")
