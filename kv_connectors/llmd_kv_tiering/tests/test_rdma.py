# Copyright 2026 The llm-d Authors.
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
"""Peer fetch through the RDMA helper, with NIXL on whatever transport UCX finds.

On a workstation UCX uses shared memory or TCP. The test checks the protocol
and the data path, not RDMA hardware.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from llmd_kv_tiering import hints as kvhints
from test_hints import action, envelope, stored_on_a
from test_vllm_tier import expected, job, make_tier, primary, run, wait_for
from vllm.v1.kv_offload.base import LookupResult, ReqContext

pytest.importorskip("nixl._api")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rdma_node(daemon_factory):
    """A daemon with an RDMA helper, and one engine tier on it."""
    made: list = []

    def node(with_helper: bool = True, **tier_kw):
        helper_sock = Path(f"/tmp/kvtier-rdma-{os.getpid()}-{len(made)}-{time.monotonic_ns()}.sock")
        args = ["--rdma-helper-socket", str(helper_sock)] if with_helper else []
        d = daemon_factory(extra_args=args)
        arr = primary()
        tier = make_tier(d, arr, **tier_kw)  # creates the pool, which the helper attaches to
        proc = None
        if with_helper:
            env = dict(os.environ, PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llmd_kv_tiering.rdma_helper",
                    "--daemon-socket",
                    str(d.socket_path),
                    "--listen-socket",
                    str(helper_sock),
                    "--name",
                    f"helper-{len(made)}-{time.monotonic_ns()}",
                ],
                env=env,
            )
            assert wait_for(helper_sock.exists, timeout=30), "RDMA helper did not start"
        made.append((tier, proc, helper_sock))
        return d, arr, tier

    yield node
    for tier, proc, sock in made:
        tier.shutdown()
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)
        sock.unlink(missing_ok=True)


def _fetch_with_hint(src, dst, arr_dst, keys, chunks):
    ctx = ReqContext(
        req_id="r",
        kv_transfer_params={
            "kv_hints": envelope(action("kv.fetch", src.endpoint, [kvhints.router_hash(k) for k in keys]))
        },
    )
    dst.on_new_request(ctx)
    j = job(9, keys, chunks, promotion=True)
    j.req_context = ctx
    return run(dst, j, store=False)


def test_fetch_reads_peer_pool_with_rdma(rdma_node):
    _, arr_a, a = rdma_node()
    d_b, arr_b, b = rdma_node()
    keys = stored_on_a(arr_a, a, n=3)

    res = _fetch_with_hint(a, b, arr_b, keys, [4, 5, 6])
    assert res.success
    for i in range(3):
        np.testing.assert_array_equal(arr_b[4 + i], expected(10 + i))
    log = d_b.log_path.read_text()
    assert "mode=rdma" in log, log[-2000:]
    # A plain fetch keeps the source copy.
    assert a.lookup(keys[0], ReqContext(req_id="x")) is LookupResult.HIT


def test_source_without_helper_falls_back_to_tcp(rdma_node):
    _, arr_a, a = rdma_node(with_helper=False)
    d_b, arr_b, b = rdma_node()
    keys = stored_on_a(arr_a, a, n=2)

    res = _fetch_with_hint(a, b, arr_b, keys, [3, 4])
    assert res.success
    np.testing.assert_array_equal(arr_b[3], expected(10))
    np.testing.assert_array_equal(arr_b[4], expected(11))


def test_coalesce_merges_consecutive_slots():
    from llmd_kv_tiering.rdma_helper import coalesce

    assert coalesce([(0, 10), (1, 11), (2, 12), (5, 3)], 100) == [(0, 1000, 300), (500, 300, 100)]
    assert coalesce([(0, 1), (1, 0)], 100) == [(0, 100, 100), (100, 0, 100)]


def test_split_runs_balances_bytes():
    from llmd_kv_tiering.rdma_helper import split_runs

    runs = [(0, 0, 800)]
    groups = split_runs(runs, 4)
    assert [sum(n for _, _, n in g) for g in groups] == [200, 200, 200, 200]
    assert groups[1] == [(200, 200, 200)]
    groups = split_runs([(0, 0, 100), (500, 900, 300)], 2)
    assert [sum(n for _, _, n in g) for g in groups] == [200, 200]
    assert split_runs([(0, 0, 100)], 8) == [
        [(0, 0, 13)],
        [(13, 13, 13)],
        [(26, 26, 13)],
        [(39, 39, 13)],
        [(52, 52, 13)],
        [(65, 65, 13)],
        [(78, 78, 13)],
        [(91, 91, 9)],
    ]


def test_zero_copy_reads_into_engine_memory(rdma_node):
    """The destination engine reads the source pool straight into its own
    CPU tier. The blocks never enter the destination's pool."""
    _, arr_a, a = rdma_node()
    d_b, arr_b, b = rdma_node(with_helper=False, rdma_zero_copy=True)
    keys = stored_on_a(arr_a, a, n=3)

    res = _fetch_with_hint(a, b, arr_b, keys, [2, 5, 6])
    assert res.success
    np.testing.assert_array_equal(arr_b[2], expected(10))
    np.testing.assert_array_equal(arr_b[5], expected(11))
    np.testing.assert_array_equal(arr_b[6], expected(12))
    assert "fetched blocks" not in d_b.log_path.read_text()  # no pooled fetch
    assert b.lookup(keys[0], ReqContext(req_id="other")) is LookupResult.MISS
    # The source's pins were released: it can still serve and evict normally.
    assert a.lookup(keys[0], ReqContext(req_id="x")) is LookupResult.HIT


def test_zero_copy_falls_back_without_source_helper(rdma_node):
    _, arr_a, a = rdma_node(with_helper=False)
    _, arr_b, b = rdma_node(with_helper=False, rdma_zero_copy=True)
    keys = stored_on_a(arr_a, a, n=2)

    res = _fetch_with_hint(a, b, arr_b, keys, [3, 4])
    assert res.success
    np.testing.assert_array_equal(arr_b[3], expected(10))
    # The pooled path ran, so the blocks are now in b's pool.
    assert b.lookup(keys[0], ReqContext(req_id="other")) is LookupResult.HIT
