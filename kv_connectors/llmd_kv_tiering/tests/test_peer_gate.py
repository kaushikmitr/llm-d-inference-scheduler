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
"""The fetch gate against a fake manager that answers like vllm-torchtpu's."""

from __future__ import annotations

import types

import pytest
from llmd_kv_tiering.peer_gate import FetchGate, host

ME, PEER, OTHER = "10.0.0.1", "10.0.0.2", "10.0.0.3"
HOST, REMOTE = "HOST", "REMOTE"


def block(status, owner):
    return types.SimpleNamespace(status=status, raiden_id=types.SimpleNamespace(job_name=owner))


class Manager:
    """lookup() is a hit only when the store finds every sub-hash."""

    def __init__(self, local=(), peer=None):
        self.local = set(local)
        self.peer = dict(peer or {})  # sub-hash -> owner
        self._block_status = types.SimpleNamespace(REMOTE=REMOTE)
        self.global_calls = 0

    def _probe(self, sub_hashes, *, enable_global=False):
        self.global_calls += enable_global
        out = []
        for h in sub_hashes:
            if h in self.local:
                out.append((h, block(HOST, ME)))
            elif enable_global and h in self.peer:
                out.append((h, block(REMOTE, self.peer[h])))
            else:
                break
        return out

    def lookup(self, key, req_context):
        subs = [key + b"0", key + b"1"]
        return "HIT" if len(self._probe(subs, enable_global=True)) == len(subs) else "MISS"


def ctx(kind=None, endpoint=None):
    params = None
    if kind:
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
    return types.SimpleNamespace(req_id="r", kv_hints=None, kv_transfer_params=params, _state={})


def on_peer(owner=PEER):
    return Manager(peer={b"k0": owner, b"k1": owner})


def test_no_hint_means_no_registry_lookup():
    m = on_peer()
    FetchGate(m)
    assert m.lookup(b"k", ctx()) == "MISS"
    assert m.global_calls == 0


def test_hint_naming_the_owner_allows_the_read():
    m = on_peer()
    gate = FetchGate(m)
    assert m.lookup(b"k", ctx("kv.fetch", f"tcp://{PEER}:17771")) == "HIT"
    assert gate.allowed == 2


def test_hint_naming_another_peer_is_a_miss():
    m = on_peer(OTHER)
    gate = FetchGate(m)
    assert m.lookup(b"k", ctx("kv.fetch", f"tcp://{PEER}:17771")) == "MISS"
    assert gate.refused == 1


def test_a_prefetch_hint_does_not_allow_a_read():
    m = on_peer()
    FetchGate(m)
    assert m.lookup(b"k", ctx("kv.prefetch", PEER)) == "MISS"


def test_local_blocks_hit_without_a_hint():
    m = Manager(local={b"k0", b"k1"})
    FetchGate(m)
    assert m.lookup(b"k", ctx()) == "HIT"


def test_local_then_hinted_peer_blocks_hit():
    m = Manager(local={b"k0"}, peer={b"k1": PEER})
    FetchGate(m)
    assert m.lookup(b"k", ctx()) == "MISS"
    assert m.lookup(b"k", ctx("kv.fetch", PEER)) == "HIT"


def test_always_mode_keeps_the_base_behavior():
    m = on_peer(OTHER)
    FetchGate(m, "always")
    assert m.lookup(b"k", ctx()) == "HIT"


def test_probes_outside_a_lookup_are_untouched():
    m = on_peer()
    FetchGate(m)
    assert len(m._probe([b"k0", b"k1"], enable_global=True)) == 2
    assert m._probe([b"k0"]) == []


def test_hints_are_parsed_once_per_request():
    m = on_peer()
    FetchGate(m)
    c = ctx("kv.fetch", PEER)
    m.lookup(b"k", c)
    c.kv_transfer_params = None  # a later call must use the cached answer
    assert m.lookup(b"k", c) == "HIT"


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        FetchGate(on_peer(), "sometimes")


def test_host():
    assert host("tcp://10.0.0.1:17771") == "10.0.0.1"
    assert host("10.0.0.1") == "10.0.0.1"
