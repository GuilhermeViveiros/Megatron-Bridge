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

from megatron.bridge.models.euro_vl.configuration_euro_vl import EuroVLConfig
from megatron.bridge.models.euro_vl.euro_vl_bridge import EuroVLBridge
from megatron.bridge.models.euro_vl.euro_vl_processor import EuroVLProcessor
from megatron.bridge.models.euro_vl.euro_vl_provider import EuroVLModelProvider

# Standalone Gemma2-backbone experiment (Tower-Plus-2B). Imported here only so its
# bridge/AutoConfig registration happens on package import; nothing above depends on it.
from megatron.bridge.models.euro_vl.gemma_euro_vl_bridge import GemmaEuroVLBridge
from megatron.bridge.models.euro_vl.gemma_euro_vl_hf import (
    GemmaEuroVLConfig,
    GemmaEuroVLForConditionalGeneration,
    GemmaEuroVLProcessor,
)
from megatron.bridge.models.euro_vl.gemma_euro_vl_provider import GemmaEuroVLModelProvider
from megatron.bridge.models.euro_vl.modeling_euro_vl import EuroVLModel, EuroVLProjector
from megatron.bridge.models.euro_vl.modeling_euro_vl_hf import (
    EuroVLForConditionalGeneration,
    EuroVLMultiModalProjector,
)
from megatron.bridge.models.euro_vl.utils import compute_moonvit_visual_tokens


__all__ = [
    "EuroVLBridge",
    "EuroVLConfig",
    "EuroVLForConditionalGeneration",
    "EuroVLModel",
    "EuroVLModelProvider",
    "EuroVLMultiModalProjector",
    "EuroVLProcessor",
    "EuroVLProjector",
    "GemmaEuroVLBridge",
    "GemmaEuroVLConfig",
    "GemmaEuroVLForConditionalGeneration",
    "GemmaEuroVLModelProvider",
    "GemmaEuroVLProcessor",
    "compute_moonvit_visual_tokens",
]
