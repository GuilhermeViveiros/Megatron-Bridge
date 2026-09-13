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

"""Shared schema and I/O helpers for the EuroVL captioning eval harness."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

MODALITIES = ("image", "multi_image", "video")

EVALS_ROOT = Path(__file__).resolve().parent
DATA_DIR = EVALS_ROOT / "data"
MEDIA_DIR = DATA_DIR / "media"
RESULTS_DIR = EVALS_ROOT / "results"


@dataclass
class EvalItem:
    """One eval-set sample, common across image / multi_image / video modalities.

    Args:
        id: Stable unique identifier, e.g. ``"sharegpt4o_0001"``.
        modality: One of ``MODALITIES``.
        media: Local file paths. A single path for ``image``/``video``, N ordered
            paths for ``multi_image``.
        prompt: Instruction/question given to the model.
        reference: Optional weak reference caption from the source dataset, used
            only as a hint for the judge (not treated as ground truth).
        source: Name of the originating HF dataset, for traceability.
    """

    id: str
    modality: str
    media: list[str]
    prompt: str
    reference: Optional[str] = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def write_jsonl(items: list[EvalItem], out_path: str | Path) -> None:
    """Write eval items to a JSONL manifest, creating parent dirs as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    logger.info("Wrote %d items to %s", len(items), out_path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json_records(records: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write a list of dicts to a JSONL file (one record per line)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), out_path)
