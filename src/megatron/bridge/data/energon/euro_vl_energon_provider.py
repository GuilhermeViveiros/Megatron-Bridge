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
``r>1`` up-samples (repeat), ``r=0`` excludes, ``r=1`` is one natural epoch. The per-source
step share is::

    step_share_i = (r_i * size_i) / Σ_j (r_j * size_j)

Energon's native ``weight`` is a size-independent sampling proportion, so repeat-factor
semantics are realized by setting ``weight_i = r_i * size_i`` (energon then samples each
source with probability ``step_share_i``).

.. note::
    ``train_iters`` is **not** auto-derived from the blend. With fill-to-``seq_length``
    packing each training step consumes a variable number of samples (≈ ``seq_length`` /
    avg-sample-length), so a sample-count → step estimate is ill-defined here. Set
    ``cfg.train.train_iters`` explicitly at launch. A packing-aware epochs→train_iters
    convenience (and a wall-clock estimate) is a tracked follow-up.

The repeat factors come from ``mixture_file`` — a YAML of ``key: r`` entries, optionally
grouped by category for readability. The file is the explicit list (**opt-in**): only the
datasets it names are used, and a dataset you omit/comment is **excluded** (``r=0`` also
drops one explicitly). ``key`` matches a dataset name (leaf dir), a path relative to
``root``, or a category prefix (applies to all datasets beneath it). A **missing** file
falls back to using every discovered dataset at ``r=1`` (one epoch each)::

    image/captioning:
      cc12m: 0.3
      coco-caption: 1.5
      sharegpt4o: 2.0
"""

import hashlib
import logging
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Optional

import yaml

from megatron.bridge.data.energon.energon_provider import EnergonProvider


logger = logging.getLogger(__name__)

# ANSI colour for the blend/train-samples summary so it stands out in the launch log.
_BLUE = "\033[94m"
_RESET = "\033[0m"

_MAX_REPEAT_FACTOR = 4.0  # InternVL uses r in (0, 4]; we also allow r=0 to exclude.


@dataclass(kw_only=True)
class EuroVLEnergonProvider(EnergonProvider):
    """EnergonProvider that builds a repeat-factor weighted blend from ``root`` + ``mixture_file``."""

    # Root directory holding prepared energon datasets (dirs with a ``.nv-meta`` subdir).
    root: str
    # YAML file of per-dataset repeat factors (optionally grouped by category).
    mixture_file: str
    # Energon fill-to-seq_length packing toggle (default on). When True, uses
    # ``packing_buffer_size`` and requires ``micro_batch_size==1`` (asserted in the base).
    # Set ``dataset.pack_to_seq_length=false`` to train without packing.
    pack_to_seq_length: bool = True
    # Where to write the generated metadataset YAML (defaults to ``root``).
    metadataset_dir: Optional[str] = None

    def _discover_datasets(self) -> dict[str, str]:
        """Map dataset name -> absolute path, for every prepared dataset (``.nv-meta``) under root.

        Datasets are keyed by their leaf dir name. When the SAME leaf name appears under two
        different paths (e.g. ``image/gui/leopard_mind2web`` vs ``multiimage/gui/leopard_mind2web``,
        or ``image/doc/sujet_finance`` vs ``image/ocr/sujet_finance``), the collision is
        disambiguated by using the FULL relative path with separators replaced by ``_``
        (``image_gui_leopard_mind2web``, ``image_doc_sujet_finance``, …) — unique by construction.
        Unique leaf names are left bare, so existing mixtures keep matching by plain name; a
        disambiguated dataset is targeted in the mixture by that prefixed name or by its relative
        path (both handled in ``_resolve_repeat_factors``).
        """
        if not self.root:
            raise ValueError("EuroVLEnergonProvider.root must be set (the energon-data directory).")
        root_abs = os.path.abspath(self.root)
        paths = [
            os.path.abspath(dirpath)
            for dirpath, dirnames, _ in os.walk(self.root)
            if ".nv-meta" in dirnames
        ]
        if not paths:
            raise ValueError(f"No prepared energon datasets (.nv-meta) found under {self.root}.")

        leaf_counts: dict[str, int] = {}
        for p in paths:
            leaf = os.path.basename(p)
            leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

        found: dict[str, str] = {}
        for p in paths:
            leaf = os.path.basename(p)
            if leaf_counts[leaf] > 1:
                # Full relative path (always unique) -> flatten to an underscore-joined name.
                name = os.path.relpath(p, root_abs).replace(os.sep, "_")
            else:
                name = leaf
            if name in found:
                raise ValueError(
                    f"Duplicate dataset key {name!r} under {self.root} even after path "
                    f"disambiguation ({found[name]} vs {p}); rename one dataset directory."
                )
            found[name] = p
        return found

    @staticmethod
    def _dataset_size(path: str) -> int:
        """Number of **train-split** samples of a prepared energon dataset.

        Counts rows in ``.nv-meta/index.sqlite`` whose shard is listed under
        ``split_parts.train`` in ``split.yaml`` — i.e. the samples actually trained on (val/test
        shards are excluded). Shard ``shard-00004.tar`` maps to ``tar_file_id`` by its numeric
        index. Falls back to the full index count if ``split.yaml`` is missing or the
        shard→id mapping is inconsistent (train+val+test != total), which is logged.
        """
        meta = os.path.join(path, ".nv-meta")
        idx = os.path.join(meta, "index.sqlite")
        if not os.path.exists(idx):
            raise FileNotFoundError(f"{idx} not found — is {path} prepared (energon prepare)?")
        con = sqlite3.connect(f"file:{idx}?mode=ro", uri=True)
        try:
            total = int(con.execute("select count(*) from samples").fetchone()[0])
            split_path = os.path.join(meta, "split.yaml")
            if not os.path.exists(split_path):
                return total

            def _count(shards) -> int:
                ids = [int(m.group(1)) for s in (shards or []) if (m := re.search(r"(\d+)", os.path.basename(s)))]
                if not ids:
                    return 0
                q = "select count(*) from samples where tar_file_id in (%s)" % ",".join("?" * len(ids))
                return int(con.execute(q, ids).fetchone()[0])

            parts = (yaml.safe_load(open(split_path)) or {}).get("split_parts", {}) or {}
            train, val, test = _count(parts.get("train")), _count(parts.get("val")), _count(parts.get("test"))
            if train <= 0 or train + val + test != total:
                logger.warning(
                    "Train-split count inconsistent for %s (train+val+test=%d != total=%d); using total.",
                    os.path.basename(path),
                    train + val + test,
                    total,
                )
                return total
            return train
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
        """Resolve the mixture_file into {absolute_path: repeat_factor}.

        The mixture file is the explicit source of truth (opt-in): only datasets it lists are
        used, each with its repeat factor ``r`` — ``r=0`` discards, ``0<r<1`` subsamples,
        ``r>1`` oversamples (effective samples = ``r * size``). Datasets **not listed are not
        in the blend** (omit/comment a dataset to drop it). A **missing** mixture file falls
        back to using every discovered dataset at ``r=1``.
        """
        by_relpath = {os.path.relpath(p, os.path.abspath(self.root)): p for p in datasets.values()}
        factors: dict[str, float] = {}  # built only from the file; unlisted datasets stay out

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

        if os.path.exists(self.mixture_file):
            with open(self.mixture_file) as f:
                for key, r in self._flatten_mixture_yaml(yaml.safe_load(f) or {}).items():
                    apply(key, r)
            dropped = sorted(name for name, p in datasets.items() if p not in factors)
            if dropped:
                logger.info("Mixture excludes %d unlisted dataset(s): %s", len(dropped), ", ".join(dropped))
        else:
            logger.warning("mixture_file %s not found; using all discovered datasets at r=1.", self.mixture_file)
            factors = {p: 1.0 for p in datasets.values()}
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
        # Atomic write: every DP rank generates this identical file concurrently. Writing in
        # place ("w" truncates to zero bytes first) leaves a microsecond window where a peer
        # rank's energon loader reads an empty file -> yaml.safe_load returns None ->
        # "'NoneType' object is not iterable". Write to a unique temp file then os.replace
        # (atomic rename) so readers only ever see the complete old or new file, never empty.
        text = yaml.safe_dump(doc, sort_keys=False)
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f".euro_vl_blend_{digest}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, out_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return out_path

    def build_datasets(self, context):
        """Generate the repeat-factor blend metadataset, point ``path`` at it, then defer to base."""
        # Gate energon fill-to-seq_length packing on the explicit toggle. When on, ensure a
        # buffer size is set (default 256); when off, disable packing so the base provider
        # skips the MBS==1 assert and trains on unpacked (padded) sequences.
        if self.pack_to_seq_length:
            if self.packing_buffer_size is None:
                self.packing_buffer_size = 256
        else:
            self.packing_buffer_size = None

        blend = self._resolve_blend(self._discover_datasets())
        total = sum(eff for *_, eff in blend)
        self.path = self._write_metadataset(blend)
        lines = [
            f"  r={r:g}  size={size:,}  eff={eff:,.0f}  step%={100 * eff / total:5.1f}  {os.path.basename(p)}"
            for p, r, size, eff in blend
        ]

        # Aggregate samples / effective-samples / step-share by category (the dataset's parent
        # path relative to root, e.g. "image/captioning", "image/ocr").
        root_abs = os.path.abspath(self.root)
        cat: dict[str, list[float]] = {}  # category -> [raw_size, eff, n_datasets]
        for p, _, size, eff in blend:
            category = os.path.dirname(os.path.relpath(p, root_abs)) or "."
            agg = cat.setdefault(category, [0.0, 0.0, 0.0])
            agg[0] += size
            agg[1] += eff
            agg[2] += 1
        total_size = sum(size for _, _, size, _ in blend)
        cat_lines = [
            f"  {c}: {int(n)} dataset(s)  size={int(s):,} ({100 * s / total_size:4.1f}% of data)"
            f"  eff={e:,.0f}  step%={100 * e / total:5.1f}"
            for c, (s, e, n) in sorted(cat.items())
        ]

        # Worst-case (1 doc/pack) train_iters to cover the effective blend N times — a safe
        # OVER-estimate under packing (packs hold ≥1 doc, so you finish sooner). Handy for
        # sizing cfg.train.train_iters; energon does not auto-derive it (see module docstring).
        gbs = self.global_batch_size
        if gbs:
            iters_line = (
                f"Suggested train_iters (worst-case guaranteed pass, GBS={gbs}): "
                f"1 epoch={math.ceil(total / gbs):,}  2 epochs={math.ceil(2 * total / gbs):,}  "
                f"3 epochs={math.ceil(3 * total / gbs):,}"
            )
        else:
            iters_line = "Suggested train_iters: global_batch_size unset — cannot estimate."

        # Log the blend summary once (global rank 0) — every rank builds the blend, but the
        # table is identical, so avoid N duplicate copies in the launch log.
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(
                _BLUE
                + "EuroVL blend (InternVL repeat factors): %d dataset(s) in %d categor(ies), "
                "total_train_samples=%d, total_eff=%.0f -> %s\n%s\n"
                "By dataset (size = TRAIN-split samples):\n%s\nBy category:\n%s"
                + _RESET,
                len(blend),
                len(cat),
                int(total_size),
                total,
                self.path,
                iters_line,
                "\n".join(lines),
                "\n".join(cat_lines),
            )
        return super().build_datasets(context)
