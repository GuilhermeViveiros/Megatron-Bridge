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

"""EuroVL Energon provider with a dynamic, weighted multi-source blend.

Generates a Megatron-Energon ``MetadatasetV2`` at runtime from a directory tree of
prepared datasets plus a ``mixture`` spec, so the data mix is fully controllable from
the launch command (CLI overrides land after recipe build, hence runtime generation).

Layout expected under ``root`` (any depth)::

    energon-data/image/captioning/cc3m/.nv-meta
    energon-data/image/captioning/coco-caption/.nv-meta
    energon-data/image/vqa/a-okvqa/.nv-meta

``mixture`` is a comma-separated ``key=weight`` spec. ``key`` matches, in order:
its dataset name (leaf dir, e.g. ``cc3m``), its path relative to ``root``
(``image/captioning/cc3m``), or a **category prefix** (``image/captioning``) which
expands to every dataset beneath it at the given weight. Weights are relative
sampling proportions (energon normalizes them). An empty ``mixture`` selects **every**
discovered dataset at weight 1.0 (i.e. all data, equal blend).

Examples::

    dataset.mixture=""                              # all datasets, equal weight
    dataset.mixture="cc3m=0.5,coco-caption=0.3"     # only these two, weighted
    dataset.mixture="image/captioning=1.0"          # all captioning datasets
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

import yaml

from megatron.bridge.data.energon.energon_provider import EnergonProvider


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EuroVLEnergonProvider(EnergonProvider):
    """EnergonProvider that builds a weighted blend metadataset from ``root`` + ``mixture``."""

    # Root directory holding prepared energon datasets (dirs with a ``.nv-meta`` subdir).
    root: str = ""
    # Comma-separated ``key=weight`` spec; empty selects all datasets at equal weight.
    mixture: str = ""
    # Where to write the generated metadataset YAML. Defaults to a content-hashed file
    # under ``root`` (idempotent across ranks). Override if ``root`` is not writable.
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

    def _resolve_mixture(self, datasets: dict[str, str]) -> list[tuple[str, float]]:
        """Resolve the mixture spec into a list of (absolute_path, weight)."""
        if not self.mixture.strip():
            return [(path, 1.0) for path in sorted(datasets.values())]

        # name -> path and relpath -> path, for key matching.
        by_relpath = {os.path.relpath(p, os.path.abspath(self.root)): p for p in datasets.values()}
        selected: dict[str, float] = {}
        for item in self.mixture.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Mixture item {item!r} must be 'key=weight'.")
            key, w = item.split("=", 1)
            key, weight = key.strip(), float(w)
            if key in datasets:  # leaf dataset name
                selected[datasets[key]] = weight
            elif key in by_relpath:  # relative path to a dataset
                selected[by_relpath[key]] = weight
            else:  # category prefix -> expand to all datasets beneath it
                prefix = key.strip("/") + "/"
                matched = [p for rel, p in by_relpath.items() if rel.startswith(prefix)]
                if not matched:
                    raise ValueError(
                        f"Mixture key {key!r} matched no dataset or category under {self.root}. "
                        f"Available datasets: {sorted(datasets)}"
                    )
                for p in matched:
                    selected[p] = weight
        return sorted(selected.items())

    def _write_metadataset(self, blend: list[tuple[str, float]]) -> str:
        """Write a MetadatasetV2 YAML for the blend and return its path."""
        refs = [{"path": path, "weight": weight} for path, weight in blend]
        doc = {
            "__module__": "megatron.energon",
            "__class__": "MetadatasetV2",
            "splits": {"train": {"blend": refs}, "val": {"blend": refs}},
        }
        out_dir = self.metadataset_dir or self.root
        os.makedirs(out_dir, exist_ok=True)
        # Content-hashed name so concurrent ranks write identical bytes (idempotent).
        digest = hashlib.sha1(yaml.safe_dump(doc, sort_keys=True).encode()).hexdigest()[:12]
        out_path = os.path.join(out_dir, f"euro_vl_blend_{digest}.metadataset.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        return out_path

    def build_datasets(self, context):
        """Generate the blend metadataset, point ``path`` at it, then defer to EnergonProvider."""
        datasets = self._discover_datasets()
        blend = self._resolve_mixture(datasets)
        self.path = self._write_metadataset(blend)
        logger.info(
            "EuroVL blend: %d dataset(s) -> %s\n%s",
            len(blend),
            self.path,
            "\n".join(f"  weight={w:g}  {p}" for p, w in blend),
        )
        return super().build_datasets(context)
