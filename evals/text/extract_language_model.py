"""Extract the language model of an HF VLM checkpoint into a standalone ``*ForCausalLM`` directory.

The text track evaluates VLMs as plain LLMs. With no image, M-RoPE positions have t=h=w, which is identical to 1D
RoPE, so the extracted model computes exactly what the VLM computes on text-only input (verified for Qwen3-VL-2B:
text-only logits max|diff| 0.0). ``mrope_*`` keys are dropped from the rope config so vLLM treats the model as an
ordinary 1D-RoPE model.

Works from safetensors plus ``config.json`` (the VLM class is never instantiated). Supported layouts:
  * EuroVL export (``EuroVLForConditionalGeneration``): ``language_model.model.*``, ``language_model.lm_head.weight``
  * Qwen3-VL (``Qwen3VLForConditionalGeneration``): ``model.language_model.*``, ``lm_head.weight`` (tied)
  * Kimi-VL (``KimiVLForConditionalGeneration``): ``language_model.model.*``, DeepSeek-V3 MoE text backbone

Two modes:
  * dense backbones (llama, qwen3): the ``*ForCausalLM`` is built and loaded with a strict key check, then saved.
  * ``deepseek_v3`` (Kimi-VL): rename-only. Tensors are copied shard by shard under the original per-expert
    DeepSeek names, which vLLM loads natively. This avoids materializing a 16B MoE on the login node and
    transformers' different fused-expert layout.

Run on the login node:
  python evals/text/extract_language_model.py --src <hf dir> --out <out dir> [--tokenizer <dir>] [--text_model_type deepseek_v3]
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM


def _preserve_legacy_tokenizer_flag(source_dir: str, output_dir: str) -> None:
    """Restore tokenizer_config.json["legacy"] in `output_dir` to match `source_dir`.

    ``save_pretrained()`` silently drops this key (verified against transformers==5.8.1: the
    live ``tokenizer.legacy`` attribute is correct in memory both before and after the drop, but
    save_pretrained() never consults it when writing). For a legacy SentencePiece tokenizer
    (e.g. EuroLLM), losing it changes how chat-turn role headers tokenize right after a special
    token (``▁assistant`` -> ``ass``+``istant``), even though tokenizer.json's own normalizer/
    pre_tokenizer fields look unchanged. Call this right after ``save_pretrained()``.

    Kept as a standalone copy (not imported from ``megatron.bridge.utils``) because this script
    runs in the lightweight text-eval env, which does not have ``megatron.bridge`` installed.
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


CAUSAL_LM = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    "qwen3_vl_text": (Qwen3Config, Qwen3ForCausalLM),
}
RENAME_ONLY = {"deepseek_v3": "DeepseekV3ForCausalLM"}

# (source key prefix, target key prefix), tried in order
KEY_PREFIXES = [
    ("language_model.model.", "model."),
    ("language_model.lm_head.", "lm_head."),
    ("model.language_model.", "model."),
    ("lm_head.", "lm_head."),
]


def rename(key: str) -> str | None:
    for src_prefix, dst_prefix in KEY_PREFIXES:
        if key.startswith(src_prefix):
            return dst_prefix + key[len(src_prefix) :]
    return None


def clean_rope(text: dict) -> dict:
    rope = dict(text.pop("rope_parameters", None) or text.pop("rope_scaling", None) or {})
    rope = {k: v for k, v in rope.items() if not k.startswith("mrope")}
    rope.setdefault("rope_type", "default")
    # Older configs keep rope_theta top-level; transformers 5.x folds it into rope_parameters. Never guess a default.
    if "rope_theta" not in rope:
        if "rope_theta" not in text:
            raise ValueError("text_config has no rope_theta (neither top-level nor in rope_parameters/rope_scaling)")
        rope["rope_theta"] = text["rope_theta"]
    return rope


def extract_dense(src: str, out: str, full_config: dict, model_type: str) -> str:
    text = dict(full_config["text_config"])
    config_cls, model_cls = CAUSAL_LM[model_type]
    rope = clean_rope(text)
    allowed = config_cls().to_dict()
    kwargs = {
        k: v for k, v in text.items() if k in allowed and k not in ("rope_parameters", "rope_scaling", "model_type")
    }
    kwargs["tie_word_embeddings"] = full_config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
    config = config_cls(**kwargs)
    config.rope_parameters = rope

    weights = {}
    for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            new = rename(key)
            if new is not None:
                weights[new] = tensor
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    model = model_cls(config).to(torch.bfloat16)
    result = model.load_state_dict(weights, strict=False)
    missing = [k for k in result.missing_keys if not (config.tie_word_embeddings and k == "lm_head.weight")]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Key mismatch: missing={missing[:10]} unexpected={result.unexpected_keys[:10]}")
    if config.tie_word_embeddings:
        model.tie_weights()
    model.save_pretrained(out)
    return f"{model_cls.__name__} ({model_type}, tied={config.tie_word_embeddings}, rope={rope}) with {len(weights)} tensors"


def extract_rename_only(src: str, out: str, full_config: dict, model_type: str) -> str:
    text = dict(full_config["text_config"])
    rope_scaling = text.get("rope_scaling")
    if rope_scaling and any(k.startswith("mrope") for k in rope_scaling):
        raise ValueError("rename-only mode does not handle M-RoPE rope_scaling")
    text["model_type"] = model_type
    text["architectures"] = [RENAME_ONLY[model_type]]
    text.setdefault("tie_word_embeddings", full_config.get("tie_word_embeddings", False))

    weight_map, n = {}, 0
    for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        name = os.path.basename(shard)
        tensors = {}
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                new = rename(key)
                if new is not None:
                    tensors[new] = f.get_tensor(key)
        if not tensors:
            continue
        save_file(tensors, os.path.join(out, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in tensors})
        n += len(tensors)
    if not any(k.startswith("lm_head.") for k in weight_map) and not text["tie_word_embeddings"]:
        raise RuntimeError("No lm_head.* tensors found for an untied model")
    total = sum(os.path.getsize(os.path.join(out, s)) for s in set(weight_map.values()))
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        open(os.path.join(out, "model.safetensors.index.json"), "w"),
        indent=2,
    )
    json.dump(text, open(os.path.join(out, "config.json"), "w"), indent=2)
    return f"{RENAME_ONLY[model_type]} ({model_type}, rename-only) with {n} tensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="HF VLM directory (config.json + safetensors)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=None, help="Directory with tokenizer files (default: --src)")
    parser.add_argument(
        "--text_model_type", default=None, help="Override when text_config has no model_type (Kimi-VL: deepseek_v3)"
    )
    args = parser.parse_args()

    full_config = json.load(open(os.path.join(args.src, "config.json")))
    model_type = args.text_model_type or full_config["text_config"].get("model_type")
    if model_type not in CAUSAL_LM and model_type not in RENAME_ONLY:
        raise ValueError(
            f"Unsupported text backbone {model_type!r}; supported: {sorted(CAUSAL_LM) + sorted(RENAME_ONLY)}"
        )

    os.makedirs(args.out, exist_ok=True)
    if model_type in RENAME_ONLY:
        summary = extract_rename_only(args.src, args.out, full_config, model_type)
    else:
        summary = extract_dense(args.src, args.out, full_config, model_type)

    tok_dir = args.tokenizer or args.src
    trust = model_type in RENAME_ONLY  # Kimi ships a custom tokenizer class
    AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=trust).save_pretrained(args.out)
    _preserve_legacy_tokenizer_flag(tok_dir, args.out)
    for extra in ("chat_template.jinja", "generation_config.json", "tiktoken.model", "tokenization_moonshot.py"):
        path = os.path.join(tok_dir, extra)
        if os.path.exists(path) and not os.path.exists(os.path.join(args.out, extra)):
            shutil.copy(path, args.out)
    print(f"{summary} -> {args.out}")


if __name__ == "__main__":
    main()
