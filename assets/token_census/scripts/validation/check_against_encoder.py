#!/usr/bin/env python3
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

"""Check the census against the REAL training task encoder, sample by sample.

For the first K samples of the first shard of each target dataset, compares
  - census: ``estimate_token_budget.count_sample()`` (header-only media math + templated text), and
  - real:   ``EuroVLTaskEncoder._cook()`` + ``encode_sample()`` (real decode, real processor), with
            ``seq_length`` / ``max_num_images`` set huge so nothing is truncated or skipped.
Total tokens, vision tokens (``<image>``/``<video>`` ids in ``input_ids``) and answer tokens (positions
with ``loss_mask > 0``) must all match exactly.

Run inside the container:
    python assets/token_census/scripts/validation/check_against_encoder.py \
        --data-root $EUROVL_DATA_ROOT --tokenizer $EUROVL_HF
"""

import argparse
import json
import os
import sys
import tarfile
import traceback
from pathlib import Path

from megatron.energon import SkipSample


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "census"))
import estimate_token_budget as etb  # noqa: E402

from megatron.bridge.data.energon.euro_vl_task_encoder import EuroVLTaskEncoder  # noqa: E402


# (modality, dataset path under the modality folder, samples to check): every sample format --
# image, multi-image, flat and message-tree video, grounding / pointing / temporal grounding, text,
# and the code / math categories whose answer tokens the README reports.
TARGETS = [
    ("image", "captioning/cc12m", 15),
    ("image", "knowledge/culturalground_oe", 15),
    ("image", "ocr/chartnet_csv", 10),
    ("image", "general_qa/rsvqa_hr", 10),
    ("image", "ocr/cc_ocr_multi_lan", 10),
    ("image", "doc/bigdocs_wikitq", 5),
    ("image", "grounding/refcoco_ground", 10),
    ("image", "pointing/refcoco_point", 10),
    ("image", "code/webcode2m_new", 10),
    ("image", "code/chartnet_code", 10),
    ("image", "code/datikz_union", 10),
    ("image", "code/websight_new", 5),
    ("image", "math/finevision_mavis_math_rule_geo", 10),
    ("image", "math/visualwebinstruct_onevision", 10),
    ("image", "math/r1_vision_stratos_17k", 5),
    ("multiimage", "chart/molmo2_table", 5),
    ("multiimage", "general_qa/doclingmatix", 5),
    ("multiimage", "captioning/multi_synth_cc12m_cc3m_grit", 5),
    ("multiimage", "pointing/mi_o365", 5),
    ("multiimage", "math/mv_math", 5),
    ("multiimage", "math/visualwebinstruct", 5),
    ("video", "captioning/activitynet_new", 3),
    ("video", "spatial/zechen_clevrer_multilingual", 3),
    ("video", "embodied_reasoning/alfred_multilingual", 3),
    ("video", "captioning/eurovideolm_packed", 2),
    ("video", "general_qa/molmo2_capqa_packed", 2),
    ("video", "general_qa/eurovideolm_qa_packed", 2),
    ("video", "temporal-grounding/hacs_packed", 2),
    ("video", "temporal-grounding/ava_temporal_grounding", 2),
    ("video", "grounding/sav_grounding", 3),
    ("video", "grounding/sav_tracking", 2),
    ("text", "chat/euroblocks", 20),
    ("text", "code/euroblocks", 20),
    ("text", "math/euroblocks", 15),
]

BASE_KEYS = {"__restore_key__": (), "__subflavor__": None, "__subflavors__": {}, "__sources__": ()}


def first_samples(modality: str, shard: Path, k: int) -> list[tuple[str, dict, dict, list[bytes]]]:
    """(key, json, {webdataset part name: bytes}, [media bytes]) for the first k samples of a shard."""
    iterate = {
        "image": etb._iter_image_samples,
        "multiimage": etb._iter_multiimage_samples,
        "video": etb._iter_video_samples,
        "text": etb._iter_text_samples,
    }[modality]
    out = []
    with tarfile.open(shard) as tf:
        for media, json_member in iterate(tf):
            key = json_member.name[: -len(".json")]
            obj = json.loads(tf.extractfile(json_member).read())
            raws, parts = [], {}
            for m in [media] if modality == "video" else list(media):
                raw = tf.extractfile(m).read()
                raws.append(raw)
                parts[m.name[len(key) + 1 :]] = raw  # e.g. jpg / img0.jpg / mp4
            out.append((key, obj, parts, raws))
            if len(out) >= k:
                break
    return out


def main() -> None:
    """Compare census and encoder counts on every target; exit non-zero on any mismatch or error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=os.environ.get("EUROVL_DATA_ROOT"), help="$EUROVL_DATA_ROOT")
    parser.add_argument("--tokenizer", default=os.environ.get("EUROVL_HF"), help="model HF export ($EUROVL_HF)")
    args = parser.parse_args()
    if args.data_root is None or args.tokenizer is None:
        raise SystemExit("--data-root and --tokenizer (or $EUROVL_DATA_ROOT / $EUROVL_HF) are required")

    etb._worker_init(args.tokenizer)
    processor = etb._processor
    encoder = EuroVLTaskEncoder(processor=processor, seq_length=10**7, max_num_images=10**6)
    tok = processor.tokenizer
    media_ids = {tok.convert_tokens_to_ids(processor.image_token), tok.convert_tokens_to_ids(processor.video_token)}

    total = mismatches = errors = 0
    answers = [0, 0]  # census, real
    for modality, rel, k in TARGETS:
        path = Path(args.data_root) / modality / rel
        shards = sorted(path.glob("shard[-_]*.tar")) or sorted(path.glob("*.tar"))
        if not shards:
            print(f"!! {modality}/{rel}: no shards")
            errors += 1
            continue
        n_ok = n_bad = 0
        for key, obj, parts, raws in first_samples(modality, shards[0], k):
            total += 1
            try:
                c = etb.count_sample(modality, json.loads(json.dumps(obj)), raws)
                census = (c["text"] + c["vision"], c["vision"], c["answer"])
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  CENSUS-ERR {modality}/{rel} {key}: {type(e).__name__}: {e}")
                continue
            try:
                encoded = encoder.encode_sample(encoder._cook({"__key__": key, **BASE_KEYS, "json": obj, **parts}))
                ids = encoded.input_ids
                vision = sum(int((ids == i).sum()) for i in media_ids)
                real = (int(ids.shape[-1]), vision, int((encoded.loss_mask > 0).sum()))
            except SkipSample:
                print(f"  REAL-SKIP  {modality}/{rel} {key} (encoder raised SkipSample)")
                continue
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  REAL-ERR   {modality}/{rel} {key}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
                continue
            answers[0] += census[2]
            answers[1] += real[2]
            if census == real:
                n_ok += 1
            else:
                n_bad += 1
                mismatches += 1
                print(f"  MISMATCH {modality}/{rel} {key}: census (total, vision, answer)={census} real={real}")
        print(f"{'OK ' if n_bad == 0 else 'BAD'} {modality}/{rel}: {n_ok} match, {n_bad} mismatch", flush=True)
    print(
        f"\nSUMMARY: {total} samples, {mismatches} mismatches, {errors} errors; "
        f"answer tokens census={answers[0]} real={answers[1]}"
    )
    if mismatches or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
