# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str = ""  # may be generated at runtime by a subclass (e.g. EuroVLEnergonProvider)
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    task_encoder: Optional[Any] = None
    # Enable batch-level online sequence packing (vlm_step in-batch packer).
    pack_sequences_in_batch: bool = False
    # Enable energon fill-to-seq_length packing: buffers this many encoded samples and
    # packs each training sequence up to seq_length (task encoder must implement the
    # select_samples_to_pack / pack_selected_samples hooks). None disables.
    packing_buffer_size: Optional[int] = None
    # Max samples loaded from a tar shard at once (energon); None loads the whole shard.
    max_samples_per_sequence: Optional[int] = None

    def build_datasets(self, context: DatasetBuildContext):
        assert self.path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"
        # The two packing modes are mutually exclusive: energon packing already yields
        # sequences pre-packed to seq_length with cu_seqlens; the in-batch packer would
        # re-concatenate and corrupt those boundaries.
        if self.packing_buffer_size is not None:
            assert not self.pack_sequences_in_batch, (
                "packing_buffer_size (energon fill-to-seq_length packing) is mutually exclusive "
                "with pack_sequences_in_batch (vlm_step in-batch packing)."
            )
            assert self.micro_batch_size == 1, (
                "energon packing yields one packed sequence per batch; set "
                f"train.micro_batch_size=1 (got {self.micro_batch_size}). THD/CP requires it."
            )
        if (
            self.pack_sequences_in_batch
            and self.task_encoder is not None
            and hasattr(self.task_encoder, "pack_sequences")
        ):
            self.task_encoder.pack_sequences = True
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            packing_buffer_size=self.packing_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            pg_collection=context.pg_collection,
        )
        return (
            iter(dataset.train_dataloader()),
            iter(dataset.val_dataloader()),
            iter(dataset.val_dataloader()),
        )
