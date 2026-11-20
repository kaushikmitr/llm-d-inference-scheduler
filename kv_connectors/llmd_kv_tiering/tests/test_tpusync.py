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
"""The TPU Sync connector against a fake store that follows the pin rules
documented for tpu_sync's KVCacheStore."""

from __future__ import annotations

import types
from collections import Counter

import pytest

pytest.importorskip("vllm")

from llmd_kv_tiering import tpusync_connector as tc  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: E402

BLOCK = 16
ME, PEER = "10.0.0.1", "10.0.0.2"


class Status:
    def __init__(self, name):
        self.name = name


class Rid:
    def __init__(self, pool, replica, data, idx):
        self.job_name, self.job_replica_id, self.data_name, self.data_replica_idx = pool, replica, data, idx


class Slice:
    def __init__(self, rid, host_block_id, device_block_id, status):
        self.raiden_id, self.device_block_id, self.status = rid, device_block_id, status


class FakeStore:
    """Local entries, a registry view of one peer, and counted pins."""

    def __init__(self, me):
        self.me = me
        self.entries: dict[bytes, str] = {}
        self.pins: Counter = Counter()
        self.peer: dict[bytes, str] = {}  # key -> owner host
        self.loads: list[tuple[bytes, int, bool]] = []  # key, device block, local
        self.saves: list[bytes] = []
        self.pushes: list[tuple[bytes, str]] = []
        self.fail_loads: set[bytes] = set()
        self.fail_pushes = False
        self.device: dict[int, bytes] = {}
        self.raiden_controller_address = "127.0.0.1:1"
        self.store_server_address = "127.0.0.1:2"

    def lookup(self, keys, enable_global, pin_found):
        out = []
        for k in keys:
            if k in self.entries:
                if pin_found:
                    self.pins[k] += 1
                out.append((k, Slice(self.me, -1, -1, Status(self.entries[k]))))
            elif enable_global and k in self.peer:
                out.append((k, Slice(Rid("p", self.peer[k], "d", 0), -1, -1, Status("REMOTE"))))
            else:
                break
        return out

    def insert(self, keys, slices, on_host):
        for k in keys:
            self.entries.setdefault(k, "HBM")
            self.pins[k] += 1
        return True

    def release(self, keys):
        for k in keys:
            assert self.pins[k] > 0, "released a block that was not pinned"
            self.pins[k] -= 1

    def save(self, keys, dst):
        for k in keys:
            assert self.pins[k] > 0, "saved a block that was not pinned"
            if dst is None:
                assert self.entries[k] == "HBM"
                self.saves.append(k)
            else:
                assert self.entries[k] in ("HOST", "HOST_AND_HBM")
                self.pushes.append((k, dst.job_replica_id))
        return True

    def load(self, keys, a, b=None):
        blocks = a if b is None else b
        for k, d in zip(keys, blocks):
            if b is None:
                assert self.pins[k] > 0, "loaded a local block that was not pinned"
            self.loads.append((k, d, b is None))
        return True

    def poll_load_status(self):
        done, failed = [], []
        for k, d, local in self.loads:
            if k in self.fail_loads:
                failed.append(k)
                continue
            done.append(k)
            self.device[d] = k
            if local:
                self.pins[k] -= 1
        self.loads = []
        return done, failed, []

    def poll_save_status(self):
        done, failed = [], []
        for k in self.saves:
            self.entries[k] = "HOST_AND_HBM"
            self.pins[k] -= 1
            done.append(k)
        for k, _ in self.pushes:
            if self.fail_pushes:
                failed.append(k)
            else:
                self.pins[k] -= 1
                done.append(k)
        self.saves, self.pushes = [], []
        return done, failed, [], [], []


def vllm_config(extra):
    return types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=BLOCK, cache_dtype="auto"),
        model_config=types.SimpleNamespace(model="m", dtype="bf16"),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config=extra),
    )


class Engine:
    """Drives both connector roles the way vLLM's scheduler and runner do."""

    def __init__(self, monkeypatch, **extra):
        monkeypatch.setattr(tc, "_runtime", None)
        monkeypatch.setattr(tc, "_settle_device", lambda runner: None)
        cfg = vllm_config({"tpusync_ip": ME, "tpusync_registry": "reg:1", **extra})
        self.sched = tc.RaidenOffloadConnector(cfg, KVConnectorRole.SCHEDULER, None)
        self.worker = tc.RaidenOffloadConnector(cfg, KVConnectorRole.WORKER, None)
        rt = self.rt = self.sched._rt
        assert self.worker._rt is rt
        status = types.SimpleNamespace(HBM=Status("HBM"))
        rt.native = types.SimpleNamespace(RaidenId=Rid, RaidenBlockId=Slice, BlockStatus=status)
        rt.rid = Rid(rt.cfg.pool, ME, "d", 0)
        self.store = rt.store = FakeStore(rt.rid)
        rt.runner = object()
        self.key = self.sched._key

    def step(self):
        self.worker.start_load_kv(None)
        send, recv = self.worker.get_finished(set())
        return send or set(), recv or set(), self.worker.get_block_ids_with_load_errors()


def request(req_id, n_blocks, hint=None):
    params = None
    if hint is not None:
        kind, endpoint = hint
        peer = "source" if kind in ("kv.fetch", "kv.evict_on_fetch") else "destination"
        params = {
            "kv_hints": {
                "protocol_version": "0.1",
                "message_id": "m",
                "actions": [
                    {
                        "action_id": "a",
                        "action_type": kind,
                        "action_version": "1.0",
                        "payload": {f"{peer}_control_endpoint": endpoint, "block_hashes": [1]},
                    }
                ],
            }
        }
    return types.SimpleNamespace(
        request_id=req_id,
        block_hashes=[b"%s-%03d" % (b"h", i) for i in range(n_blocks)],
        kv_transfer_params=params,
        kv_hints=None,
    )


def blocks(ids):
    return types.SimpleNamespace(get_block_ids=lambda: (list(ids),))


def finish(e, req, ids):
    delay, _ = e.sched.request_finished(req, list(ids))
    if delay:
        send, _, _ = e.step()
        assert send == {req.request_id}
    return delay


def test_save_then_local_load(monkeypatch):
    e = Engine(monkeypatch)
    a = request("a", 3)
    assert e.sched.get_num_new_matched_tokens(a, 0) == (0, False)
    assert finish(e, a, [10, 11, 12]) is True
    assert e.rt.stats.saved == 3
    assert not +e.store.pins

    # Saved blocks are not saved again.
    assert e.sched.request_finished(request("a2", 3), [20, 21, 22]) == (False, None)

    b = request("b", 4)
    assert e.sched.get_num_new_matched_tokens(b, 0) == (3 * BLOCK, True)
    e.sched.update_state_after_alloc(b, blocks([30, 31, 32, 33]), 3 * BLOCK)
    _, recv, bad = e.step()
    assert recv == {"b"} and not bad
    assert [e.store.device[i] for i in (30, 31, 32)] == [e.key(h) for h in b.block_hashes[:3]]
    assert not +e.store.pins


def test_blocks_the_engine_holds_are_skipped(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 4), [1, 2, 3, 4])
    b = request("b", 4)
    assert e.sched.get_num_new_matched_tokens(b, 2 * BLOCK) == (2 * BLOCK, True)
    e.sched.update_state_after_alloc(b, blocks([50, 51, 52, 53]), 2 * BLOCK)
    e.step()
    assert sorted(e.store.device) == [52, 53]
    assert not +e.store.pins


def test_unscheduled_match_releases_its_pins(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 2), [1, 2])
    b = request("b", 2)
    e.sched.get_num_new_matched_tokens(b, 0)
    assert sum(e.store.pins.values()) == 2
    e.sched.get_num_new_matched_tokens(b, 0)  # matched again before it was scheduled
    assert sum(e.store.pins.values()) == 2
    e.sched.update_state_after_alloc(b, blocks([5, 6]), 0)
    assert not +e.store.pins


def test_peer_blocks_need_a_fetch_hint_naming_the_peer(monkeypatch):
    e = Engine(monkeypatch)
    plain = request("r", 3)
    for h in plain.block_hashes:
        e.store.peer[e.key(h)] = PEER
    assert e.sched.get_num_new_matched_tokens(plain, 0) == (0, False)
    other = request("r", 3, hint=("kv.fetch", "tcp://10.9.9.9:17771"))
    assert e.sched.get_num_new_matched_tokens(other, 0) == (0, False)

    hinted = request("r", 3, hint=("kv.fetch", f"tcp://{PEER}:17771"))
    assert e.sched.get_num_new_matched_tokens(hinted, 0) == (3 * BLOCK, True)
    e.sched.update_state_after_alloc(hinted, blocks([7, 8, 9]), 3 * BLOCK)
    assert e.store.loads == [(e.key(h), d, False) for h, d in zip(hinted.block_hashes, (7, 8, 9))]
    _, recv, bad = e.step()
    assert recv == {"r"} and not bad
    assert not +e.store.pins


def test_always_mode_reads_any_peer(monkeypatch):
    e = Engine(monkeypatch, tpusync_global_lookup="always")
    r = request("r", 2)
    for h in r.block_hashes:
        e.store.peer[e.key(h)] = PEER
    assert e.sched.get_num_new_matched_tokens(r, 0) == (2 * BLOCK, True)


def test_local_then_peer_blocks_load_in_two_calls(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 2), [1, 2])
    r = request("r", 4, hint=("kv.fetch", PEER))
    for h in r.block_hashes[2:]:
        e.store.peer[e.key(h)] = PEER
    assert e.sched.get_num_new_matched_tokens(r, 0) == (4 * BLOCK, True)
    e.sched.update_state_after_alloc(r, blocks([40, 41, 42, 43]), 4 * BLOCK)
    assert [local for _, _, local in e.store.loads] == [True, True, False, False]
    _, recv, _ = e.step()
    assert recv == {"r"}
    assert not +e.store.pins


def test_failed_blocks_are_reported_invalid(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 3), [1, 2, 3])
    b = request("b", 3)
    e.store.fail_loads = {e.key(b.block_hashes[1])}
    e.sched.get_num_new_matched_tokens(b, 0)
    e.sched.update_state_after_alloc(b, blocks([60, 61, 62]), 3 * BLOCK)
    _, recv, bad = e.step()
    assert recv == {"b"} and bad == {61}
    assert not +e.store.pins


def test_second_request_waits_for_a_block_that_is_loading(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 2), [1, 2])
    b, c = request("b", 2), request("c", 2)
    e.sched.get_num_new_matched_tokens(b, 0)
    e.sched.update_state_after_alloc(b, blocks([5, 6]), 2 * BLOCK)
    assert e.sched.get_num_new_matched_tokens(c, 0) == (None, False)
    e.step()
    assert e.sched.get_num_new_matched_tokens(c, 0) == (2 * BLOCK, True)
    e.sched.update_state_after_alloc(c, blocks([7, 8]), 0)
    assert not +e.store.pins


def test_request_that_ends_while_loading_frees_after_the_load(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 2), [1, 2])
    b = request("b", 2)
    e.sched.get_num_new_matched_tokens(b, 0)
    e.sched.update_state_after_alloc(b, blocks([5, 6]), 2 * BLOCK)
    assert e.sched.request_finished(b, [5, 6]) == (True, None)
    assert not e.rt.plans
    _, recv, _ = e.step()
    assert recv == {"b"}


def test_prefetch_hint_offers_saved_blocks_to_the_peer(monkeypatch):
    e = Engine(monkeypatch)
    a = request("a", 3, hint=("kv.prefetch", f"tcp://{PEER}:17771"))
    assert e.sched.request_finished(a, [1, 2, 3]) == (True, None)
    assert e.sched.has_pending_push_work()
    send, _, _ = e.step()  # saves start and end; the push starts
    assert send == {"a"}
    assert e.store.pushes == [(e.key(h), PEER) for h in a.block_hashes]
    assert e.sched.has_pending_push_work()
    e.step()
    assert e.rt.stats.pushed == 3 and not e.sched.has_pending_push_work()
    assert not +e.store.pins


def test_refused_push_releases_its_pins(monkeypatch):
    e = Engine(monkeypatch)
    finish(e, request("a", 2), [1, 2])
    e.store.fail_pushes = True
    again = request("a2", 2, hint=("kv.prefetch", PEER))
    assert e.sched.request_finished(again, [3, 4]) == (False, None)
    e.step()
    assert e.rt.stats.push_failed == 2
    assert not +e.store.pins


def test_host_of_endpoint():
    assert tc._host("tcp://10.0.0.1:17771") == "10.0.0.1"
    assert tc._host("10.0.0.1:17771") == "10.0.0.1"
    assert tc._host("10.0.0.1") == "10.0.0.1"
    assert tc._host("tcp://[fd00::1]:17771") == "fd00::1"
