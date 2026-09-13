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

"""LLM-as-judge absolute scoring for generated captions.

Supports Anthropic and OpenAI as the judge provider, called via stdlib HTTP
(no ``anthropic``/``openai`` SDK dependency). The judge sees the source
media, the prompt, the generated caption, and (if available) a weak
reference caption from the source dataset, and returns an absolute 1-5
rubric score plus a short rationale.

Example:
  ANTHROPIC_API_KEY=... uv run python -m evals.judge \\
    --captions evals/results/v1/image_captions.jsonl \\
    --manifest evals/data/image_eval.jsonl \\
    --model_tag v1 --judge_provider anthropic --judge_model claude-sonnet-5
"""

import argparse
import base64
import concurrent.futures
import json
import logging
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evals.common import RESULTS_DIR, read_jsonl, write_json_records


logger = logging.getLogger(__name__)

RUBRIC_AXES = (
    "helpfulness",
    "correctness",
    "faithfulness",
    "completeness",
    "relevance",
    "coherence",
    "readability",
    "overall",
)
MAX_JUDGE_FRAMES = 4

_RUBRIC_INSTRUCTIONS = f"""You are grading a vision-language model's response to a prompt, given the source
media shown to you. Score the response on each axis from 1 (worst) to 5 (best):

- helpfulness: how useful is this caption to someone who cannot see the media themselves?
- correctness: are the factual claims in the response correct, given what is actually shown?
- faithfulness: does the response stick to content actually present in the media, without
  inventing objects, attributes, or events that are not visually grounded?
- completeness: does the response cover the salient subjects/actions in the media, without
  omitting obviously important content?
- relevance: does the response actually address the prompt (describing the media), without
  drifting into irrelevant or off-topic content?
- coherence: is the response internally consistent, without logical gaps or self-contradiction
  (e.g. repeating or contradicting its own earlier claims)?
- readability: is the response clearly written, well-formed English at an appropriate level of
  detail (not garbled, telegraphic, or degenerately repetitive)?
- overall: your overall quality judgment.

Respond with ONLY a JSON object with keys {list(RUBRIC_AXES)} (integers 1-5) and a "rationale"
key (one or two sentences). Do not include any other text."""

# Used when the judge model has no vision (e.g. a local text-only judge). Without the source
# media, "correctness"/"faithfulness"/"completeness" can only be graded against the weak
# reference caption, so scores from this mode are strictly a fallback for harness testing
# without a vision-capable judge, not a substitute for the vision-grounded rubric above.
_TEXT_ONLY_RUBRIC_INSTRUCTIONS = f"""You are grading a vision-language model's response to a prompt about an
image/video you cannot see yourself. Judge using only the prompt, the response, and (if given) a weak
reference caption. Score the response on each axis from 1 (worst) to 5 (best):

- helpfulness: how useful would this caption be to someone who cannot see the media themselves?
- correctness: does the response seem plausible and consistent with the reference caption, if given?
- faithfulness: score 5 if the response makes no oddly specific claims that contradict the
  reference caption; lower for claims that look fabricated or contradictory.
- completeness: does the response seem to cover what the reference caption covers, without
  large gaps?
- relevance: does the response actually address the prompt (describing the media), without
  drifting into irrelevant or off-topic content?
- coherence: is the response internally consistent, without logical gaps, repetition, or
  self-contradiction?
- readability: is the response clearly written, well-formed English?
- overall: your overall quality judgment.

Respond with ONLY a JSON object with keys {list(RUBRIC_AXES)} (integers 1-5) and a "rationale"
key (one or two sentences). Do not include any other text."""

_PAIRWISE_INSTRUCTIONS = """You are comparing two vision-language model responses to the same prompt, given the
source media shown to you. You will see "Response A" and "Response B" — these are anonymized labels, not a hint
about quality or order. Decide which response is more accurate and faithful to the media (fewer hallucinated
objects/attributes/events, better coverage of what is actually shown), using fluency/readability only as a
tie-breaker when accuracy is otherwise equal. Respond "tie" only if you genuinely cannot distinguish them on
accuracy grounds.

Respond with ONLY a JSON object with keys "winner" (one of "A", "B", "tie") and "rationale" (one or two
sentences explaining the deciding factor). Do not include any other text."""


def _b64_image(path: str) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return media_type, base64.b64encode(f.read()).decode("utf-8")


def _sample_video_frames(video_path: str, num_frames: int = MAX_JUDGE_FRAMES) -> list[str]:
    """Sample a few frames from a video and save them as temp JPEGs for the judge.

    Decodes directly with PyAV rather than ``qwen_vl_utils.fetch_video``'s torchvision backend,
    whose batched ``to_rgb()``/swscale call hit ``av.error.BlockingIOError: Resource temporarily
    unavailable`` on some hosts. Converts (``to_image()``) and keeps only the target frames —
    never materializes every decoded frame as a full-resolution PIL image at once, which under
    ``--concurrency`` (many videos decoding in parallel) OOM-killed the process at ~130GB RSS.
    """
    import av

    from evals.datasets._common import save_image

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total = stream.frames or sum(1 for _ in container.decode(stream))
    if not total:
        raise ValueError(f"No frames found in {video_path}")

    indices = sorted({int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(min(num_frames, total))})
    target = set(indices)

    out_dir = Path(video_path).parent / "_judge_frames" / Path(video_path).stem
    frame_paths: dict[int, str] = {}
    last_path: str | None = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx in target:
                last_path = str(save_image(frame.to_image(), out_dir / f"frame_{idx:02d}.jpg"))
                frame_paths[idx] = last_path
                if len(frame_paths) == len(target):
                    break
    if len(frame_paths) < len(target):
        # stream.frames can over-report the real decodable frame count (corrupt trailing data,
        # B-frame reordering quirks); reuse the last frame actually decoded for any index the
        # stream ended before reaching, rather than raise on the (already-good) rest.
        if last_path is None:
            raise ValueError(f"No frames decoded from {video_path}")
        logger.warning(
            "%s: stream.frames overreported (%d), only decoded up to index %d; reusing last frame for %s",
            video_path,
            total,
            max(frame_paths, default=-1),
            sorted(target - frame_paths.keys()),
        )
        for idx in target - frame_paths.keys():
            frame_paths[idx] = last_path
    return [frame_paths[i] for i in indices]


class LLMJudge:
    """Thin HTTP client for absolute-rubric caption judging via Anthropic, OpenAI, or a local
    OpenAI-compatible server (e.g. vLLM serving a vision-language model; pass ``vision=False``
    for a text-only local judge, which falls back to ``_TEXT_ONLY_RUBRIC_INSTRUCTIONS`` and
    never sends media)."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        vision: bool = True,
    ):
        if provider not in ("anthropic", "openai", "local"):
            raise ValueError(f"Unknown judge_provider: {provider!r}")
        self.provider = provider
        self.model = model
        # "localhost" resolves via getaddrinfo, which on some compute nodes returns an IPv6
        # address the node doesn't actually support -- urllib.request (unlike curl) does not
        # fall back to IPv4, so every judge call fails with "OSError: [Errno 97] Address family
        # not supported by protocol" even though the health-check curl call succeeds. Use the
        # literal IPv4 loopback address to sidestep the resolution ambiguity entirely.
        self.base_url = base_url or "http://127.0.0.1:8000/v1"
        # anthropic/openai are always vision-capable APIs; for "local" this controls whether the
        # served model can actually see media (a vision-language model) or is text-only, in which
        # case we fall back to _TEXT_ONLY_RUBRIC_INSTRUCTIONS and never attach media.
        self.vision = vision if provider == "local" else True
        if provider == "local":
            self.api_key = api_key or os.environ.get("LOCAL_JUDGE_API_KEY", "EMPTY")
        else:
            env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            self.api_key = api_key or os.environ.get(env_var)
            if not self.api_key:
                raise ValueError(f"Set {env_var} or pass --judge_api_key.")

    def score(self, media_paths: list[str], prompt: str, caption: str, reference: str | None) -> dict[str, Any]:
        """Score one generated caption. Retries once on malformed JSON.

        ``media_paths`` is ignored for the ``local`` (text-only) provider.
        """
        user_text = f'Prompt given to the model: "{prompt}"\n\nModel response: "{caption}"'
        if reference:
            user_text += f'\n\nWeak reference caption (may be imperfect, use as a hint only): "{reference}"'

        for attempt in range(2):
            raw = self._call(media_paths, user_text)
            parsed = self._parse(raw)
            if parsed is not None:
                return parsed
            logger.warning("Malformed judge response on attempt %d, retrying: %.200s", attempt, raw)
        return {axis: None for axis in RUBRIC_AXES} | {"rationale": "judge response could not be parsed"}

    def _parse(self, raw: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not all(axis in parsed for axis in RUBRIC_AXES):
            return None
        return parsed

    def compare(self, media_paths: list[str], prompt: str, caption_a: str, caption_b: str) -> dict[str, Any]:
        """Pairwise-compare two captions for the same media. Retries once on malformed JSON.

        Returns ``{"winner": "A"|"B"|"tie", "rationale": str}`` (winner ``None`` if unparseable).
        """
        user_text = (
            f'Prompt given to both models: "{prompt}"\n\nResponse A: "{caption_a}"\n\nResponse B: "{caption_b}"'
        )
        for attempt in range(2):
            raw = self._call(media_paths, user_text, system_prompt=_PAIRWISE_INSTRUCTIONS)
            parsed = self._parse_pairwise(raw)
            if parsed is not None:
                return parsed
            logger.warning("Malformed pairwise response on attempt %d, retrying: %.200s", attempt, raw)
        return {"winner": None, "rationale": "judge response could not be parsed"}

    def _parse_pairwise(self, raw: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if parsed.get("winner") not in ("A", "B", "tie"):
            return None
        return parsed

    def _call(self, media_paths: list[str], user_text: str, *, system_prompt: str = _RUBRIC_INSTRUCTIONS) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(media_paths, user_text, system_prompt)
        if self.provider == "local":
            return self._call_local(media_paths, user_text, system_prompt)
        return self._call_openai(media_paths, user_text, system_prompt)

    def _call_anthropic(self, media_paths: list[str], user_text: str, system_prompt: str) -> str:
        content = []
        for path in media_paths:
            media_type, data = _b64_image(path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model,
            "max_tokens": 512,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        response = self._post(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        return "".join(block.get("text", "") for block in response.get("content", []))

    def _call_openai(self, media_paths: list[str], user_text: str, system_prompt: str) -> str:
        content = [{"type": "text", "text": user_text}]
        for path in media_paths:
            media_type, data = _b64_image(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})

        payload = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        response = self._post(
            "https://api.openai.com/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return response["choices"][0]["message"]["content"]

    def _call_local(self, media_paths: list[str], user_text: str, system_prompt: str) -> str:
        """Call a local OpenAI-compatible server (e.g. vLLM), with media attached if the served
        model is vision-capable (``self.vision``), else a text-only rubric prompt."""
        if self.vision:
            content = [{"type": "text", "text": user_text}]
            for path in media_paths:
                media_type, data = _b64_image(path)
                content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
            user_content: Any = content
        else:
            if system_prompt == _RUBRIC_INSTRUCTIONS:
                system_prompt = _TEXT_ONLY_RUBRIC_INSTRUCTIONS
            user_content = user_text

        payload = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # Some local judge models (e.g. Qwen3-style) default to a chain-of-thought "thinking"
            # mode that burns max_tokens on reasoning before ever emitting the JSON rubric,
            # causing malformed-response retries. Disable it for fast, deterministic JSON output.
            "chat_template_kwargs": {"enable_thinking": False},
            # Without this, the model occasionally emits a JSON array (or other near-JSON) instead
            # of an object, especially on multi-image inputs — vLLM's guided decoding enforces
            # valid JSON syntax (not our specific keys, so _parse's key check still matters).
            "response_format": {"type": "json_object"},
        }
        response = self._post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return response["choices"][0]["message"]["content"]

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Judge API call failed ({e.code}): {e.read().decode('utf-8', 'replace')}") from e


def judge_captions(
    manifest: list[dict], captions: list[dict], judge: LLMJudge, model_tag: str, *, concurrency: int = 1
) -> list[dict]:
    """Score every generated caption in ``captions`` against its manifest media.

    ``concurrency`` dispatches multiple judge calls in parallel threads — a local vLLM server
    with ``--max-num-seqs`` > 1 batches concurrent requests internally, so sequential (the
    default) leaves most of its throughput on the table. Order is preserved in the output
    regardless of concurrency.
    """
    manifest_by_id = {item["id"]: item for item in manifest}
    prepared = []
    for record in captions:
        item = manifest_by_id.get(record["id"])
        if item is None:
            logger.warning("No manifest entry for id=%s, skipping", record["id"])
            continue
        prepared.append((record, item))

    def _score_one(record: dict, item: dict) -> dict:
        # A leaked/incomplete <think> reasoning trace (base-model habit surfacing on some
        # inputs, see generate_captions.py's hf_cached_decode*) means the caption is not
        # real judgeable output -- score it 0 directly rather than sending it to the judge.
        if "<think>" in record["caption"] or "</think>" in record["caption"]:
            scores = {axis: 0 for axis in RUBRIC_AXES} | {"rationale": "caption contains a leaked <think> tag"}
            return {"id": record["id"], "modality": item["modality"], "model_tag": model_tag, **scores}
        media = item["media"]
        if item["modality"] == "video" and judge.vision:
            media = _sample_video_frames(media[0])
        try:
            scores = judge.score(media, record["prompt"], record["caption"], item.get("reference"))
        except Exception:
            # One oversized/malformed item (e.g. a video whose sampled frames push the request
            # past the server's context limit) must not lose every other item's judge scores.
            logger.exception("Judge call failed for id=%s, recording as unscored", record["id"])
            scores = {axis: None for axis in RUBRIC_AXES} | {"rationale": "judge call raised an exception"}
        return {"id": record["id"], "modality": item["modality"], "model_tag": model_tag, **scores}

    if concurrency <= 1:
        return [_score_one(record, item) for record, item in prepared]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda pair: _score_one(*pair), prepared))


def pairwise_judge_captions(
    manifest: list[dict],
    captions_a: list[dict],
    captions_b: list[dict],
    judge: LLMJudge,
    tag_a: str,
    tag_b: str,
    *,
    seed: int = 0,
    concurrency: int = 1,
) -> list[dict]:
    """Pairwise-compare two checkpoints' captions on every item they share.

    A/B order is randomized per item (seeded, so reruns are reproducible) to cancel out any
    position bias in the judge — the returned records always report ``winner_tag`` (the actual
    checkpoint tag that won, or "tie"/None), not the raw A/B label. See ``judge_captions`` for
    what ``concurrency`` > 1 does; order is preserved in the output regardless.
    """
    import random

    rng = random.Random(seed)
    manifest_by_id = {item["id"]: item for item in manifest}
    captions_b_by_id = {r["id"]: r for r in captions_b}
    prepared = []
    for record_a in captions_a:
        item_id = record_a["id"]
        item = manifest_by_id.get(item_id)
        record_b = captions_b_by_id.get(item_id)
        if item is None or record_b is None:
            logger.warning("No manifest/caption_b entry for id=%s, skipping", item_id)
            continue
        prepared.append((record_a, record_b, item, rng.random() < 0.5))

    def _compare_one(record_a: dict, record_b: dict, item: dict, swap: bool) -> dict:
        item_id = record_a["id"]
        media = item["media"]
        if item["modality"] == "video" and judge.vision:
            media = _sample_video_frames(media[0])

        cap_a, cap_b = (
            (record_b["caption"], record_a["caption"]) if swap else (record_a["caption"], record_b["caption"])
        )
        label_tag = {"A": tag_b if swap else tag_a, "B": tag_a if swap else tag_b}

        try:
            result = judge.compare(media, record_a["prompt"], cap_a, cap_b)
        except Exception:
            logger.exception("Pairwise judge call failed for id=%s, recording as unscored", item_id)
            result = {"winner": None, "rationale": "judge call raised an exception"}

        winner = result.get("winner")
        winner_tag = "tie" if winner == "tie" else label_tag.get(winner)
        return {
            "id": item_id,
            "modality": item["modality"],
            "tag_a": tag_a,
            "tag_b": tag_b,
            "swapped": swap,
            "winner_tag": winner_tag,
            "rationale": result.get("rationale"),
        }

    if concurrency <= 1:
        return [_compare_one(*args) for args in prepared]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda args: _compare_one(*args), prepared))


def main(args) -> None:
    """Judge a caption file against its manifest and write per-item scores."""
    manifest = read_jsonl(args.manifest)
    judge = LLMJudge(
        args.judge_provider,
        args.judge_model,
        args.judge_api_key,
        base_url=args.judge_base_url,
        vision=not args.judge_text_only,
    )
    modality = manifest[0]["modality"] if manifest else "unknown"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "judged"

    if args.captions_b:
        captions_a = read_jsonl(args.captions)
        captions_b = read_jsonl(args.captions_b)
        results = pairwise_judge_captions(
            manifest, captions_a, captions_b, judge, args.model_tag, args.model_tag_b, concurrency=args.concurrency
        )
        out_path = out_dir / f"{modality}_{args.model_tag}_vs_{args.model_tag_b}.jsonl"
        write_json_records(results, out_path)
        return

    captions = read_jsonl(args.captions)
    scored = judge_captions(manifest, captions, judge, args.model_tag, concurrency=args.concurrency)
    out_path = out_dir / f"{modality}_{args.model_tag}_scores.jsonl"
    write_json_records(scored, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Judge generated captions with an LLM-as-judge rubric.")
    parser.add_argument("--manifest", type=str, required=True, help="Eval manifest JSONL (has media/reference).")
    parser.add_argument("--captions", type=str, required=True, help="generate_captions.py output JSONL.")
    parser.add_argument("--model_tag", type=str, required=True, help="Label for the judged checkpoint, e.g. 'v1'.")
    parser.add_argument("--judge_provider", type=str, choices=["anthropic", "openai", "local"], default="anthropic")
    parser.add_argument("--judge_model", type=str, default="claude-sonnet-5", help="Judge model id.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Overrides ANTHROPIC_API_KEY/OPENAI_API_KEY.")
    parser.add_argument(
        "--judge_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for --judge_provider local (default http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--judge_text_only",
        action="store_true",
        help="For --judge_provider local: the served model has no vision, so never attach media "
        "and fall back to the weak-reference-only rubric. Default assumes a vision-capable local judge.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Dir to write <modality>_<model_tag>_scores.jsonl into (default evals/results/judged).",
    )
    parser.add_argument(
        "--captions_b",
        type=str,
        default=None,
        help="If set, switches to pairwise mode: compares --captions (as --model_tag) against "
        "--captions_b (as --model_tag_b) on every shared id, instead of absolute scoring.",
    )
    parser.add_argument(
        "--model_tag_b", type=str, default=None, help="Label for the second checkpoint in pairwise mode."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of judge calls to dispatch in parallel threads. A local vLLM server batches "
        "concurrent requests internally, so >1 (e.g. 32) is much faster than the default sequential.",
    )
    args = parser.parse_args()
    if args.captions_b and not args.model_tag_b:
        parser.error("--captions_b requires --model_tag_b")
    main(args)
