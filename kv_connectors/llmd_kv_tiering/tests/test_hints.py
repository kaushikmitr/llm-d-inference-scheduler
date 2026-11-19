# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Router hints: parsing, and transfers between two real daemons."""

from __future__ import annotations

import numpy as np
import pytest
from llmd_kv_tiering import hints as kvhints
from test_vllm_tier import (
    CTX,
    expected,
    fill,
    job,
    key,
    make_tier,
    primary,
    run,
    wait_for,
)
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_offload.base import LookupResult, ReqContext


def action(action_type, endpoint, hashes, action_id="a1", version="1.0", **extra):
    peer = "source" if action_type in ("kv.fetch", "kv.evict_on_fetch") else "destination"
    payload = {f"{peer}_control_endpoint": endpoint, "block_hashes": list(hashes), **extra}
    return {
        "action_id": action_id,
        "action_type": action_type,
        "action_version": version,
        "payload": payload,
    }


def envelope(*actions):
    return {"protocol_version": "0.1", "message_id": "m1", "actions": list(actions)}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_all_four_actions():
    p = kvhints.parse(
        envelope(
            action("kv.fetch", "tcp://s:1", [1, 2], "f"),
            action("kv.evict_on_fetch", "tcp://s:1", [3], "e"),
            action("kv.prefetch", "tcp://d:1", [4], "p"),
            action("kv.move", "tcp://d:1", [5], "m"),
        )
    )
    assert not p.rejected
    assert [h.action_id for h in p.fetches] == ["f", "e"]
    assert [h.action_id for h in p.pushes] == ["p", "m"]
    assert [h.evict_source for h in p.hints] == [False, True, False, True]
    assert p.hints[0].block_hashes == {1, 2}


@pytest.mark.parametrize(
    "payload",
    [
        {"source_control_endpoint": "", "block_hashes": [1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": []},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [True]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [-1]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [1 << 64]},
        {"source_control_endpoint": "tcp://source:1", "block_hashes": [1], "no_retain": True},
        {"block_hashes": [1]},
    ],
)
def test_parse_rejects_invalid_payloads(payload):
    raw = {"action_id": "x", "action_type": "kv.fetch", "action_version": "1.0", "payload": payload}
    p = kvhints.parse(envelope(raw))
    assert not p.hints and len(p.rejected) == 1


def test_bad_action_does_not_sink_the_envelope():
    p = kvhints.parse(
        envelope(
            {"action_id": "u", "action_type": "kv.teleport", "action_version": "1.0", "payload": {}},
            action("kv.fetch", "tcp://s:1", [7], "ok"),
            "not-an-action",
        )
    )
    assert [h.action_id for h in p.hints] == ["ok"]
    assert {r.reason for r in p.rejected} == {"unsupported_action_type", "invalid_hint"}


def test_parse_accepts_vllm_struct():
    env = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="m",
        actions=[
            KvHintAction(
                action_id="a",
                action_type="kv.fetch",
                action_version="1.0",
                payload={"source_control_endpoint": "tcp://s:1", "block_hashes": [9]},
            )
        ],
    )
    p = kvhints.parse(env)
    assert [h.block_hashes for h in p.hints] == [frozenset({9})]


def test_parse_rejects_other_versions():
    p = kvhints.parse({**envelope(action("kv.fetch", "tcp://s:1", [9])), "protocol_version": "0.2"})
    assert not p.hints and p.rejected[0].action_type == "envelope"
    old = action("kv.fetch", "tcp://s:1", [9], version="1.1")
    p = kvhints.parse(envelope(old, action("kv.fetch", "tcp://t:1", [8], "a2")))
    assert [h.endpoint for h in p.hints] == ["tcp://t:1"]
    assert "action_version" in p.rejected[0].reason


def test_router_hash_matches_vllm():
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_offload.base import get_offload_block_hash, make_offload_key

    bh = bytes(range(32))
    k = make_offload_key(bh, 3)
    want = int.from_bytes(get_offload_block_hash(k), "big") & ((1 << 64) - 1)
    assert kvhints.router_hash(k) == want
    assert BlockHash(bh) == get_offload_block_hash(k)


# ---------------------------------------------------------------------------
# Transfers between two daemons
# ---------------------------------------------------------------------------


@pytest.fixture
def pair(daemon_factory):
    """Two nodes, each a daemon with one engine tier."""
    made = []

    def node(**kw):
        d = daemon_factory()
        arr = primary()
        t = make_tier(d, arr, **kw)
        made.append(t)
        assert t.connected and t.endpoint
        return arr, t

    yield node
    for t in made:
        t.shutdown()


def stored_on_a(arr_a, tier_a, n=3, tag0=10):
    keys = [key(i) for i in range(n)]
    for i in range(n):
        fill(arr_a, i, tag=tag0 + i)
    assert run(tier_a, job(1, keys, list(range(n))), store=True).success
    return keys


@pytest.mark.parametrize("carrier", ["kv_transfer_params", "kv_hints"])
def test_fetch_hint_pulls_from_peer(pair, carrier):
    arr_a, a = pair()
    arr_b, b = pair(opportunistic_query=False)
    keys = stored_on_a(arr_a, a)
    hinted = keys[:2]
    env = envelope(action("kv.fetch", a.endpoint, [kvhints.router_hash(k) for k in hinted]))
    if carrier == "kv_transfer_params":
        ctx = ReqContext(req_id="r1", kv_transfer_params={"kv_hints": env})
    else:
        ctx = ReqContext(
            req_id="r1",
            kv_hints=KvHintsEnvelope(
                protocol_version="0.1",
                message_id="m1",
                actions=[KvHintAction(**a_) for a_ in env["actions"]],
            ),
        )
    b.on_new_request(ctx)

    assert b.lookup(hinted[0], ctx) is LookupResult.HIT
    assert b.lookup(keys[2], ctx) is LookupResult.MISS  # not hinted, strict mode
    assert b.lookup(hinted[0], CTX) is LookupResult.MISS  # other request

    j = job(2, hinted, [5, 6], promotion=True)
    j.req_context = ctx
    assert run(b, j, store=False).success
    np.testing.assert_array_equal(arr_b[5], expected(10))
    np.testing.assert_array_equal(arr_b[6], expected(11))

    # The blocks are now in b's pool: any request hits, hint or not.
    b.on_request_finished(ctx)
    assert b.lookup(hinted[0], CTX) is LookupResult.HIT
    # A plain fetch leaves the source copy.
    assert a.lookup(hinted[0], CTX) is LookupResult.HIT


def test_evict_on_fetch_removes_source_copy(pair):
    arr_a, a = pair()
    arr_b, b = pair()
    keys = stored_on_a(arr_a, a, n=1)
    ctx = ReqContext(
        req_id="r1",
        kv_transfer_params={
            "kv_hints": envelope(action("kv.evict_on_fetch", a.endpoint, [kvhints.router_hash(keys[0])]))
        },
    )
    b.on_new_request(ctx)
    j = job(2, keys, [4], promotion=True)
    j.req_context = ctx
    assert run(b, j, store=False).success
    np.testing.assert_array_equal(arr_b[4], expected(10))
    assert wait_for(lambda: a.lookup(keys[0], CTX) is LookupResult.MISS)


def test_unreachable_source_fails_cleanly(pair):
    _, b = pair()
    k = key(1)
    ctx = ReqContext(
        req_id="r1",
        kv_transfer_params={"kv_hints": envelope(action("kv.fetch", "tcp://127.0.0.1:1", [kvhints.router_hash(k)]))},
    )
    b.on_new_request(ctx)
    assert b.lookup(k, ctx) is LookupResult.HIT  # optimistic: the load decides
    j = job(2, [k], [0], promotion=True)
    j.req_context = ctx
    res = run(b, j, store=False)
    assert not res.success and not res.successful_keys


@pytest.mark.parametrize("action_type", ["kv.prefetch", "kv.move"])
def test_push_hints_copy_to_destination(pair, action_type):
    arr_a, a = pair()
    arr_b, b = pair()
    keys = stored_on_a(arr_a, a, n=2)
    ctx = ReqContext(
        req_id="carrier",
        kv_transfer_params={
            "kv_hints": envelope(action(action_type, b.endpoint, [kvhints.router_hash(k) for k in keys]))
        },
    )
    a.on_new_request(ctx)
    assert wait_for(lambda: all(b.lookup(k, CTX) is LookupResult.HIT for k in keys))
    assert run(b, job(3, keys, [0, 1], promotion=True), store=False).success
    np.testing.assert_array_equal(arr_b[1], expected(11))
    moved = action_type == "kv.move"
    assert wait_for(lambda: all((a.lookup(k, CTX) is LookupResult.MISS) == moved for k in keys))


def test_self_source_hint_is_local(pair):
    arr_a, a = pair()
    keys = stored_on_a(arr_a, a, n=1)
    ctx = ReqContext(
        req_id="r1",
        kv_transfer_params={"kv_hints": envelope(action("kv.fetch", a.endpoint, [kvhints.router_hash(key(99))]))},
    )
    a.on_new_request(ctx)
    assert a.lookup(keys[0], ctx) is LookupResult.HIT
    assert a.lookup(key(99), ctx) is LookupResult.MISS


def test_fetch_hint_through_vllm_tiering_manager(pair, daemon_factory):
    """A cold engine B serves a request whose prefix lives on node A, using
    only the router's hint, through vLLM's own TieringOffloadingManager."""
    from test_vllm_tier import _Region
    from vllm.v1.kv_offload.base import ScheduleEndContext, make_offload_key
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    arr_a, a = pair()
    keys = [make_offload_key(f"blk{i}".encode().ljust(8, b"\0"), 0) for i in range(3)]
    for i in range(3):
        fill(arr_a, i, tag=200 + i)
    assert run(a, job(1, keys, [0, 1, 2]), store=True).success

    d_b = daemon_factory()
    arr_b = primary(num_chunks=4)
    tier_b = make_tier(d_b, arr_b)
    mgr = TieringOffloadingManager(
        primary_tier=CPUPrimaryTierOffloadingManager(num_chunks=4, mmap_region=_Region(arr_b)),
        secondary_tiers=[tier_b],
    )

    def step():
        mgr.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
        list(mgr.take_events())

    try:
        ctx = ReqContext(
            req_id="routed-to-b",
            kv_transfer_params={
                "kv_hints": envelope(action("kv.fetch", a.endpoint, [kvhints.router_hash(k) for k in keys]))
            },
        )
        mgr.on_new_request(ctx)
        assert all(mgr.lookup(k, ctx) is LookupResult.HIT_PENDING for k in keys)
        assert wait_for(
            lambda: (step(), tier_b.drain_jobs(), step()) and all(mgr.lookup(k, ctx) is LookupResult.HIT for k in keys)
        )
        rows = {arr_b[c].tobytes() for c in range(4)}
        assert all(expected(200 + i).tobytes() in rows for i in range(3))
    finally:
        tier_b.shutdown()


def test_opportunistic_fetch_with_placeholder_hash(pair):
    """llm-d's EPP sends block_hashes [0] and relies on opportunistic_query:
    every block of the request is fetchable from the named source."""
    arr_a, a = pair()
    arr_b, b = pair()
    keys = stored_on_a(arr_a, a, n=2)
    absent = key(50)  # never computed anywhere
    ctx = ReqContext(
        req_id="r1",
        kv_hints=KvHintsEnvelope(
            protocol_version="0.1",
            message_id="m",
            actions=[
                KvHintAction(
                    action_id="fetch-0",
                    action_type="kv.fetch",
                    action_version="1.0",
                    payload={"source_control_endpoint": a.endpoint, "block_hashes": [0]},
                )
            ],
        ),
    )
    b.on_new_request(ctx)
    assert all(b.lookup(k, ctx) is LookupResult.HIT for k in keys + [absent])
    j = job(2, keys + [absent], [3, 4, 5], promotion=True)
    j.req_context = ctx
    res = run(b, j, store=False)
    assert not res.success and set(res.successful_keys) == set(keys)
    np.testing.assert_array_equal(arr_b[3], expected(10))
    # The source lacked it, so it is not offered again.
    assert b.lookup(absent, ctx) is LookupResult.MISS


def test_evict_on_fetch_is_never_opportunistic(pair):
    arr_a, a = pair()
    _, b = pair()
    keys = stored_on_a(arr_a, a, n=2)
    ctx = ReqContext(
        req_id="r1",
        kv_transfer_params={
            "kv_hints": envelope(action("kv.evict_on_fetch", a.endpoint, [kvhints.router_hash(keys[0])]))
        },
    )
    b.on_new_request(ctx)
    assert b.lookup(keys[0], ctx) is LookupResult.HIT
    assert b.lookup(keys[1], ctx) is LookupResult.MISS


def test_first_hint_for_a_hash_wins(pair):
    _arr, tier = pair()
    env = envelope(
        action("kv.fetch", "tcp://first:1", [5, 6], "a1"),
        action("kv.fetch", "tcp://second:1", [6, 7], "a2"),
    )
    tier.on_new_request(ReqContext(req_id="r", kv_transfer_params={"kv_hints": env}))
    hints = tier._req_fetch["r"]
    assert {h: hints.by_hash[h].endpoint for h in (5, 6, 7)} == {
        5: "tcp://first:1",
        6: "tcp://first:1",
        7: "tcp://second:1",
    }
    # The first plain kv.fetch also covers the request's unlisted blocks.
    assert hints.opportunistic.endpoint == "tcp://first:1"
