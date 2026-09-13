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

"""HF-side classes (config, model, processor) for the Gemma2-backbone EuroVL experiment.

Thin subclasses of the stock EuroVL HF classes. The config/model pair exists to carry a
distinct ``model_type``: the Megatron bridge registry dispatches on the HF class name and
model_type, so giving this variant its own pair keeps :class:`EuroVLBridge` — which would
otherwise claim a ``euro_vl`` checkpoint and route a Gemma2 backbone down its Llama path —
completely untouched.

The stock ``EuroVLForConditionalGeneration`` already builds an arbitrary backbone through
``AutoModelForCausalLM.from_config(text_config)`` (which resolves ``gemma2`` ->
``Gemma2ForCausalLM``) and already returns ``None`` from ``_compute_position_ids`` for
non-Qwen3 backbones, letting Gemma2 use its own 1D RoPE. The only behavioural override is
vision-feature rescaling, which Gemma2's embedding normalizer makes necessary --- see
``GemmaEuroVLForConditionalGeneration._embedding_rescale``.
"""

from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import EuroVLForConditionalGeneration


# Tower-Plus-2B (Widn/Tower-Plus-2B), a Gemma2-2B derivative. Mirrors its config.json.
_TOWER_PLUS_2B_TEXT_DEFAULTS = dict(
    vocab_size=256000,
    hidden_size=2304,
    intermediate_size=9216,
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=256,
    max_position_embeddings=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    hidden_activation="gelu_pytorch_tanh",
    attn_logit_softcapping=50.0,
    final_logit_softcapping=30.0,
    query_pre_attn_scalar=224,
    sliding_window=4096,
    attention_bias=False,
    tie_word_embeddings=True,
)

# Gemma2 ships no vision tokens, but it does carry <unused0>..<unused98> at ids 7..105.
# We repurpose the first five rather than extending the vocab. Extending would give the
# vision delimiters freshly-random embeddings that then sit frozen through the whole PA
# stage (the LLM is frozen there), even though those tokens appear in every sample — and
# it would also force a resize of Gemma2's tied embedding/output matrix. The <unused*>
# slots are already real rows in the pretrained embedding and already tokenize atomically
# (they are declared added tokens), so nothing about the tokenizer has to change; the
# processor below just points at these strings instead of EuroVL's default names.
IMAGE_TOKEN = "<unused0>"
VISION_START_TOKEN = "<unused1>"
VISION_END_TOKEN = "<unused2>"
VISION_PAD_TOKEN = "<unused3>"
VIDEO_TOKEN = "<unused4>"

VISION_TOKEN_IDS = {
    IMAGE_TOKEN: 7,
    VISION_START_TOKEN: 8,
    VISION_END_TOKEN: 9,
    VISION_PAD_TOKEN: 10,
    VIDEO_TOKEN: 11,
}


class GemmaEuroVLConfig(EuroVLConfig):
    """EuroVL config with a Gemma2 (Tower-Plus-2B) text backbone."""

    model_type = "gemma_euro_vl"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projector_input_dim: int = 4608,
        projector_output_dim: int = 2304,
        image_token_id: int = VISION_TOKEN_IDS[IMAGE_TOKEN],
        vision_start_token_id: int = VISION_TOKEN_IDS[VISION_START_TOKEN],
        vision_end_token_id: int = VISION_TOKEN_IDS[VISION_END_TOKEN],
        vision_pad_token_id: int = VISION_TOKEN_IDS[VISION_PAD_TOKEN],
        video_token_id: int = VISION_TOKEN_IDS[VIDEO_TOKEN],
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        # The base class defaults text_config to EuroLLM/Llama; default to Gemma2 instead.
        # Dicts still fall through to the base class, which resolves them by model_type.
        if text_config is None:
            text_config = Gemma2Config(**_TOWER_PLUS_2B_TEXT_DEFAULTS)

        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            projector_input_dim=projector_input_dim,
            projector_output_dim=projector_output_dim,
            image_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            vision_pad_token_id=vision_pad_token_id,
            video_token_id=video_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GemmaEuroVLForConditionalGeneration(EuroVLForConditionalGeneration):
    """EuroVL (MoonViT + projector) on a Gemma2 decoder.

    Compensates for Gemma2's embedding normalizer so that inference matches how the
    projector was trained. See :meth:`get_image_features`.
    """

    config_class = GemmaEuroVLConfig
    _no_split_modules = ["MoonVitEncoderLayer", "Gemma2DecoderLayer"]

    def _embedding_rescale(self) -> float:
        """Factor cancelling Gemma2's ``sqrt(hidden_size)`` embedding normalizer.

        Gemma2 is the only EuroVL backbone that scales embeddings, and the two training
        stacks apply that scale at *different points*:

        - Megatron (training): ``EmbeddingScalingMixin`` scales the **embedding module's
          output**, and ``EuroVLModel`` scatters vision features in afterwards -- so the
          decoder sees text at ``x sqrt(H)`` and vision at ``x 1``.
        - HF (inference): vision is scattered into raw embeddings, then ``Gemma2Model``
          multiplies the **whole** ``inputs_embeds`` by ``sqrt(H)`` -- scaling vision too.

        Left uncorrected, projected features arrive ``sqrt(2304) = 48x`` larger than the
        projector was trained to emit; they swamp the residual stream and generation
        collapses into repetition loops, while *text-only* prompts stay perfectly fluent
        (no vision features to mis-scale). Dividing here restores the training-time ratio.
        This mirrors upstream HF PaliGemma, which divides image features by
        ``hidden_size**0.5`` for the same reason.
        """
        return float(self.config.text_config.hidden_size) ** -0.5

    def get_image_features(self, pixel_values, image_grid_thw):
        """Projected image tokens, pre-divided to cancel Gemma2's embedding normalizer."""
        return super().get_image_features(pixel_values, image_grid_thw) * self._embedding_rescale()

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        """Projected video tokens, pre-divided to cancel Gemma2's embedding normalizer."""
        return super().get_video_features(pixel_values_videos, video_grid_thw) * self._embedding_rescale()


class GemmaEuroVLProcessor(EuroVLProcessor):
    """EuroVL processor pointing at Gemma2's repurposed ``<unused*>`` vision tokens.

    Same pipeline as the base processor (MoonViT image/video processing, placeholder
    expansion); only the placeholder *strings* differ, mirroring how
    ``Qwen3EuroVLProcessor`` swaps in Qwen3's built-in vision tokens.
    """

    image_token = IMAGE_TOKEN
    video_token = VIDEO_TOKEN
    vision_start_token = VISION_START_TOKEN
    vision_end_token = VISION_END_TOKEN
