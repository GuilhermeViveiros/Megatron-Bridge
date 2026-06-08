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

"""EuroVL Energon provider with an InternVL-style repeat-factor data blend.

Builds a Megatron-Energon ``MetadatasetV2`` at runtime from a directory of prepared
datasets (``root``) plus a ``mixture`` of per-dataset **repeat factors** ``r``, so the
mix is fully controllable from the launch command (CLI overrides land after recipe
build, hence runtime generation).

Repeat factor (InternVL): ``r_i`` is how many epochs of dataset *i* to use, so its
**effective sample count is ``r_i * size_i``**. ``r<1`` down-samples (use a fraction),
``r>1`` up-samples (repeat), ``r=0`` excludes, ``r=1`` is one natural epoch. Because the
contribution is ``r_i * size_i``, the total is determinate and the max training steps can
be computed a priori::

    total_samples = Σ (r_i * size_i)
    train_iters   = ceil(epochs * total_samples / global_batch_size)
    step_share_i  = (r_i * size_i) / total_samples

Energon's native ``weight`` is a size-independent sampling proportion, so repeat-factor
semantics are realized by setting ``weight_i = r_i * size_i`` (energon then samples each
source with probability ``step_share_i``).

The repeat factors come from ``mixture_file`` — a YAML of ``key: r`` entries, optionally
grouped by category for readability. Every discovered dataset starts at ``r=1`` and listed
keys override it (so you only tweak a few; ``r=0`` drops one). ``key`` matches a dataset
name (leaf dir), a path relative to ``root``, or a category prefix (applies to all
datasets beneath it). A missing file leaves every dataset at ``r=1`` (one epoch each)::

    image/captioning:
      cc12m: 0.3
      coco-caption: 1.5
      sharegpt4o: 2.0
"""

import hashlib
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import yaml

from megatron.bridge.data.energon.energon_provider import EnergonProvider


logger = logging.getLogger(__name__)

_MAX_REPEAT_FACTOR = 4.0  # InternVL uses r in (0, 4]; we also allow r=0 to exclude.


@dataclass(kw_only=True)
class EuroVLEnergonProvider(EnergonProvider):
    """EnergonProvider that builds a repeat-factor weighted blend from ``root`` + ``mixture_file``."""

    # Root directory holding prepared energon datasets (dirs with a ``.nv-meta`` subdir).
    root: str
    # YAML file of per-dataset repeat factors (optionally grouped by category).
    mixture_file: str
    # Epochs over the (repeat-factor-weighted) blend; drives auto train_iters. None disables.
    epochs: Optional[float] = 1.0
    # Where to write the generated metadataset YAML (defaults to ``root``).
    metadataset_dir: Optional[str] = None

    def _discover_datasets(self) -> dict[str, str]:
        """Map dataset name (leaf dir) -> absolute path, for every prepared dataset under root."""
        if not self.root:
            raise ValueError("EuroVLEnergonProvider.root must be set (the energon-data directory).")
        found: dict[str, str] = {}
        for dirpath, dirnames, _ in os.walk(self.root):
            if ".nv-meta" in dirnames:
                name = os.path.basename(dirpath.rstrip("/"))
                if name in found:
                    raise ValueError(f"Duplicate dataset name {name!r} under {self.root}; names must be unique.")
                found[name] = os.path.abspath(dirpath)
        if not found:
            raise ValueError(f"No prepared energon datasets (.nv-meta) found under {self.root}.")
        return found

    @staticmethod
    def _dataset_size(path: str) -> int:
        """Indexed sample count of a prepared energon dataset (from .nv-meta/index.sqlite)."""
        idx = os.path.join(path, ".nv-meta", "index.sqlite")
        if not os.path.exists(idx):
            raise FileNotFoundError(f"{idx} not found — is {path} prepared (energon prepare)?")
        con = sqlite3.connect(f"file:{idx}?mode=ro", uri=True)
        try:
            return int(con.execute("select count(*) from samples").fetchone()[0])
        finally:
            con.close()

    @staticmethod
    def _flatten_mixture_yaml(data) -> dict[str, float]:
        """Flatten a (possibly category-nested) mixture YAML into {key: repeat_factor}.

        A mapping value is a category group (recurse); a numeric value is a repeat factor
        for that key (a dataset name or a category prefix).
        """
        flat: dict[str, float] = {}
        if not isinstance(data, dict):
            raise ValueError("mixture_file must contain a mapping (optionally nested by category).")
        for key, val in data.items():
            if isinstance(val, dict):
                flat.update(EuroVLEnergonProvider._flatten_mixture_yaml(val))
            else:
                flat[str(key)] = float(val)
        return flat

    def _resolve_repeat_factors(self, datasets: dict[str, str]) -> dict[str, float]:
        """Resolve the mixture_file into {absolute_path: repeat_factor} (datasets default to 1.0)."""
        factors: dict[str, float] = {p: 1.0 for p in datasets.values()}
        by_relpath = {os.path.relpath(p, os.path.abspath(self.root)): p for p in datasets.values()}

        def apply(key: str, r: float) -> None:
            if not (0.0 <= r <= _MAX_REPEAT_FACTOR):
                raise ValueError(f"repeat factor for {key!r} must be in [0, {_MAX_REPEAT_FACTOR}], got {r}")
            if key in datasets:  # leaf dataset name
                targets = [datasets[key]]
            elif key in by_relpath:  # relative path to a dataset
                targets = [by_relpath[key]]
            else:  # category prefix -> all datasets beneath it
                prefix = key.strip("/") + "/"
                targets = [p for rel, p in by_relpath.items() if rel.startswith(prefix)]
                if not targets:
                    raise ValueError(
                        f"Mixture key {key!r} matched no dataset or category under {self.root}. "
                        f"Available: {sorted(datasets)}"
                    )
            for p in targets:
                factors[p] = r

        # Apply the per-dataset repeat factors from the mixture YAML (convention:
        # energon-data/mixture.yaml). If absent, every dataset stays at r=1.
        if os.path.exists(self.mixture_file):
            with open(self.mixture_file) as f:
                for key, r in self._flatten_mixture_yaml(yaml.safe_load(f) or {}).items():
                    apply(key, r)
        else:
            logger.warning("mixture_file %s not found; using default repeat factors (r=1).", self.mixture_file)
        return factors

    def _resolve_blend(self, datasets: dict[str, str]) -> list[tuple[str, float, int, float]]:
        """Return [(path, repeat_factor, size, effective_samples)] for non-excluded datasets."""
        factors = self._resolve_repeat_factors(datasets)
        blend = []
        for path in sorted(factors):
            r = factors[path]
            if r <= 0.0:
                continue  # excluded
            size = self._dataset_size(path)
            blend.append((path, r, size, r * size))
        if not blend:
            raise ValueError("Mixture excluded every dataset (all repeat factors are 0).")
        return blend

    def compute_train_iters(self, global_batch_size: int) -> Optional[int]:
        """Auto max-steps: ceil(epochs * Σ(r_i * size_i) / global_batch_size). None if epochs unset."""
        if self.epochs is None:
            return None
        blend = self._resolve_blend(self._discover_datasets())
        total = sum(eff for _, _, _, eff in blend)
        return max(1, math.ceil(self.epochs * total / global_batch_size))

    def _write_metadataset(self, blend: list[tuple[str, float, int, float]]) -> str:
        """Write a MetadatasetV2 YAML using energon weights = r_i * size_i; return its path."""
        refs = [{"path": path, "weight": float(eff)} for path, _, _, eff in blend]
        doc = {
            "__module__": "megatron.energon",
            "__class__": "MetadatasetV2",
            "splits": {"train": {"blend": refs}, "val": {"blend": refs}},
        }
        out_dir = self.metadataset_dir or self.root
        os.makedirs(out_dir, exist_ok=True)
        digest = hashlib.sha1(yaml.safe_dump(doc, sort_keys=True).encode()).hexdigest()[:12]
        out_path = os.path.join(out_dir, f"euro_vl_blend_{digest}.metadataset.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        return out_path

    def build_datasets(self, context):
        """Generate the repeat-factor blend metadataset, point ``path`` at it, then defer to base."""
        blend = self._resolve_blend(self._discover_datasets())
        total = sum(eff for *_, eff in blend)
        self.path = self._write_metadataset(blend)
        lines = [
            f"  r={r:g}  size={size:,}  eff={eff:,.0f}  step%={100 * eff / total:5.1f}  {os.path.basename(p)}"
            for p, r, size, eff in blend
        ]
        logger.info(
            "EuroVL blend (InternVL repeat factors): %d dataset(s), total_eff=%.0f -> %s\n%s",
            len(blend),
            total,
            self.path,
            "\n".join(lines),
        )
        return super().build_datasets(context)
