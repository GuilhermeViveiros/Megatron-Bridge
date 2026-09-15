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

"""Preserve the ``legacy`` tokenizer flag across a HuggingFace ``save_pretrained()`` round-trip.

``PreTrainedTokenizerFast.save_pretrained()`` silently drops the source ``tokenizer_config.json``'s
``legacy`` key (verified against ``transformers==5.8.1``: the live ``tokenizer.legacy`` Python
attribute is correct in memory both before and after this drop, but ``save_pretrained()`` never
consults it when deciding what to write). For a tokenizer whose class builds its pre-tokenizer
differently depending on ``legacy`` at *load* time (e.g. Llama-family SentencePiece tokenizers,
which pick ``Metaspace(prepend_scheme="always")`` when ``legacy=True`` vs ``"first"`` otherwise),
losing this key changes real tokenization behavior on reload -- e.g. a chat-turn role header
right after a special token, such as ``▁assistant``, re-segments into ``ass`` + ``istant``.

This is *not* a normalizer/pre_tokenizer field swap (both source and resaved ``tokenizer.json``
correctly show ``prepend_scheme="always"`` even when broken) -- it is specifically this one
top-level ``tokenizer_config.json`` key being dropped, confirmed by patching only this key back in
and observing tokenization become byte-identical to the source again.
"""

import json
import os


def preserve_legacy_tokenizer_flag(source_dir: str, output_dir: str) -> None:
    """Restore ``tokenizer_config.json["legacy"]`` in `output_dir` to match `source_dir`.

    Call this *after* ``tokenizer.save_pretrained(output_dir)``. Only touches the file when the
    source tokenizer actually declares ``legacy`` (some tokenizer classes never set it, in which
    case there is nothing to preserve and this is a no-op) and the value actually changed.

    Args:
        source_dir: Directory of the original tokenizer that was loaded (e.g. via
            ``AutoTokenizer.from_pretrained``) before extending/resaving it.
        output_dir: Directory `save_pretrained()` just wrote to.
    """
    source_cfg_path = os.path.join(source_dir, "tokenizer_config.json")
    if not os.path.isfile(source_cfg_path):
        return
    with open(source_cfg_path) as f:
        source_legacy = json.load(f).get("legacy")
    if source_legacy is None:
        return

    output_cfg_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.isfile(output_cfg_path):
        return
    with open(output_cfg_path) as f:
        output_cfg = json.load(f)
    if output_cfg.get("legacy") == source_legacy:
        return

    output_cfg["legacy"] = source_legacy
    with open(output_cfg_path, "w") as f:
        json.dump(output_cfg, f, ensure_ascii=False)
