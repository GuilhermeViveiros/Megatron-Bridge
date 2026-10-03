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

"""Backport of the upstream megatron-energon fix for ``Sharder._split_shards``.

energon < 7 walks the shard-boundary table with an unbounded loop::

    while True:
        if end_offset <= shard_cumsums[end_index + 1]:
            break
        end_index += 1

energon splits a dataset into one slice per worker, where the worker count is
``world_size * num_workers``. When a split holds fewer samples than that, the
trailing workers get nothing and their slice starts at the very end of the data.
``np.searchsorted`` then returns the last valid index and the loop reads one past
the end of the array::

    IndexError: index 2 is out of bounds for axis 0 with size 2

An empty worker is legitimate and must simply yield no samples, so this is a
library bug, not a data problem. It is scale-dependent: the same dataset loads
fine on few ranks and fails once enough ranks are added. One rank raises while
the others block in a collective until the NCCL timeout, which buries the real
error.

Upstream fixed it in 7.x by bounding the loop. That single condition is the only
behavioural difference between 6.0.1 and 7.3.2 in the splitting path (everything
else added in 7.x is the ``subset`` feature, a no-op when ``subset is None``), so
``_split_shards`` below is the upstream function reproduced verbatim.

The patch installs itself only while the installed energon still carries the bug
— detected by running the failing case, not by comparing version strings — so it
turns into a no-op the moment the container ships a fixed energon.
"""

import logging
from typing import Generator, Optional, Sequence

import numpy as np
from megatron.energon.flavors.webdataset.sharder import Sharder


logger = logging.getLogger(__name__)

_PATCH_ATTR = "_megatron_bridge_split_shards_patched"


def _split_shards(
    cls,
    shard_cumsums: np.ndarray,
    offsets: Sequence[int],
    *,
    max_samples_per_sequence: Optional[int],
) -> Generator[Sequence[int], None, None]:
    """Upstream energon 7.x ``Sharder._split_shards``, verbatim.

    The only change against 6.0.1 is the bounded ``while`` condition below.
    """
    # Find shard idx for start
    start_index = np.searchsorted(shard_cumsums, offsets[0], side="right") - 1

    for start_offset, end_offset in zip(offsets, offsets[1:]):
        # Find shard idx for end
        end_index = start_index
        while end_index + 1 < len(shard_cumsums) and end_offset > shard_cumsums[end_index + 1]:
            end_index += 1
        if start_index == end_index:
            yield (
                *cls._split_shard(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    max_samples_per_sequence=max_samples_per_sequence,
                ),
                end_offset,
            )
        else:
            # Middle is the original shards, start and end get an offset/length
            yield (
                *(
                    cls._split_shard(
                        start_offset=start_offset,
                        end_offset=shard_cumsums[start_index + 1],
                        max_samples_per_sequence=max_samples_per_sequence,
                    )
                    if shard_cumsums[start_index + 1] > start_offset
                    else ()
                ),
                *(
                    offset
                    for inner_shard_start, inner_shard_end in zip(
                        shard_cumsums[start_index + 1 : end_index],
                        shard_cumsums[start_index + 2 : end_index + 1],
                    )
                    for offset in cls._split_shard(
                        start_offset=inner_shard_start,
                        end_offset=inner_shard_end,
                        max_samples_per_sequence=max_samples_per_sequence,
                    )
                ),
                *cls._split_shard(
                    start_offset=shard_cumsums[end_index],
                    end_offset=end_offset,
                    max_samples_per_sequence=max_samples_per_sequence,
                ),
                end_offset,
            )
        start_index = end_index


def energon_sharder_is_buggy() -> bool:
    """Return True if the installed energon raises on a worker slice at the end of a split.

    Feature detection rather than a version comparison: one shard of 10 samples and a
    worker whose slice is ``[10, 10)`` is exactly the shape that trips the unbounded loop.
    """
    try:
        list(Sharder._split_shards(np.cumsum([0, 10]), [10, 10], max_samples_per_sequence=None))
    except IndexError:
        return True
    return False


def apply_energon_sharder_fix() -> bool:
    """Install the upstream fix if needed. Idempotent; returns True if the patch is active."""
    if getattr(Sharder, _PATCH_ATTR, False):
        return True
    if not energon_sharder_is_buggy():
        return False
    Sharder._split_shards = classmethod(_split_shards)
    setattr(Sharder, _PATCH_ATTR, True)
    logger.info(
        "Applied the upstream energon fix for Sharder._split_shards: this energon raises "
        "IndexError when a worker's slice starts at the end of a split (splits smaller than "
        "world_size * num_workers). Upgrading to megatron-energon>=7 makes this a no-op."
    )
    return True
