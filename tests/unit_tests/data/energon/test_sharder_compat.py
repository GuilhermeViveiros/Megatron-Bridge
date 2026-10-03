import numpy as np
import pytest
from megatron.energon.flavors.webdataset.sharder import Sharder

from megatron.bridge.data.energon.sharder_compat import (
    apply_energon_sharder_fix,
    energon_sharder_is_buggy,
)


class _WorkerConfig:
    """shard_workers reads only these three attributes."""

    def __init__(self, rank: int, world_size: int, num_workers: int):
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers


class _ShardInfo:
    def __init__(self, count: int):
        self.count = count


def _all_worker_ranges(shard_counts, world_size, num_workers):
    """Every [start, end) range every worker of every rank will read.

    Mirrors the contract in energon's sample_loader: consecutive offsets of a
    worker delimit the sample ranges it reads.
    """
    shards = [_ShardInfo(c) for c in shard_counts]
    ranges = []
    for rank in range(world_size):
        per_worker = Sharder.shard_workers(
            shards, _WorkerConfig(rank, world_size, num_workers), max_samples_per_sequence=None
        )
        for offsets in per_worker:
            ranges.extend((int(a), int(b)) for a, b in zip(offsets, offsets[1:]) if b > a)
    return ranges


def _assert_exact_partition(shard_counts, world_size, num_workers):
    """The workers' ranges must tile [0, total) with no gap, overlap or loss."""
    total = sum(shard_counts)
    ranges = sorted(_all_worker_ranges(shard_counts, world_size, num_workers))
    cursor = 0
    for start, end in ranges:
        assert start >= cursor, f"overlap at [{start},{end}) after {cursor}"
        assert start == cursor, f"gap [{cursor},{start})"
        cursor = end
    assert cursor == total, f"covered {cursor} of {total} samples"
    assert sum(e - s for s, e in ranges) == total


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_energon_sharder_fix()


@pytest.mark.unit
def test_probe_agrees_with_actual_behaviour():
    """energon_sharder_is_buggy must reflect reality, since the patch is gated on it."""
    buggy_before_patch = energon_sharder_is_buggy()
    assert isinstance(buggy_before_patch, bool)
    # After apply() (module fixture) the sharder must never raise on the probe case.
    assert not energon_sharder_is_buggy()


@pytest.mark.unit
def test_apply_is_idempotent():
    assert apply_energon_sharder_fix() in (True, False)
    assert not energon_sharder_is_buggy()


@pytest.mark.unit
@pytest.mark.parametrize("total", [1, 2, 5, 10, 31, 32, 33, 40, 95, 128, 1000])
@pytest.mark.parametrize("world_size", [1, 4, 16, 64, 256])
def test_tiny_split_partitions_exactly_at_scale(total, world_size):
    """A split smaller than world_size * num_workers is legal: extra workers get nothing.

    This is the case energon < 7 crashed on (IndexError in _split_shards).
    """
    _assert_exact_partition([total], world_size, num_workers=8)


@pytest.mark.unit
@pytest.mark.parametrize(
    "shard_counts",
    [[10, 0, 10], [0, 10], [10, 0], [7, 13, 1, 200], [200] * 5, [1] * 37],
)
@pytest.mark.parametrize("world_size", [1, 16, 256])
def test_multi_shard_layouts_partition_exactly(shard_counts, world_size):
    """Zero-count shards and uneven shards must not break the tiling either."""
    _assert_exact_partition(shard_counts, world_size, num_workers=8)


@pytest.mark.unit
def test_empty_split_does_not_raise():
    """A split with no samples yields nothing rather than crashing one rank."""
    assert _all_worker_ranges([0], world_size=16, num_workers=8) == []


@pytest.mark.unit
def test_known_regression_case_mi_taco():
    """10 val samples on 16 ranks x 8 workers: rank 15's slice starts at the end.

    This is the exact shape that failed the 4-node full-mixture pre-flight with
    'IndexError: index 2 is out of bounds for axis 0 with size 2'.
    """
    list(Sharder._split_shards(np.cumsum([0, 10]), [10, 10], max_samples_per_sequence=None))
    _assert_exact_partition([10], world_size=16, num_workers=8)
