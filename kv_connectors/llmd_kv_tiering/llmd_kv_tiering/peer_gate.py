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
"""Gate TPU Sync peer reads on router fetch hints.

vllm-torchtpu's TPU Sync offloading manager reads a peer's blocks on every
local miss once a global registry is configured. This gate makes that read
depend on the request's ``kv.fetch`` hints: a peer's block counts as a hit
only when a hint names that peer. Local host memory and the storage tier
are not affected.

The gate wraps two methods of the manager instance, ``lookup`` and
``_probe``. ``lookup`` is where the manager decides whether a block is a
hit, and it reads the store through ``_probe``.
"""

from __future__ import annotations

import logging
from typing import Any

from llmd_kv_tiering import hints as kvhints

logger = logging.getLogger("vllm.llmd_kv_tiering.peer_gate")

MODES = ("hint", "always")
_ANY = object()


def host(endpoint: str) -> str:
    """The host of a control endpoint such as ``tcp://10.0.0.1:17771``."""
    s = endpoint.split("://", 1)[-1]
    if s.startswith("["):
        return s[1 : s.index("]")]
    return s.rsplit(":", 1)[0] if ":" in s else s


class _Peers:
    """Per-request cache slot in ReqContext's scratch space."""

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names


def hinted_peers(req_context: Any) -> frozenset[str]:
    """The peers this request's fetch hints name, parsed once per request."""
    state = getattr(req_context, "_state", None)
    if state is not None and _Peers in state:
        return state[_Peers].names
    env = kvhints.envelope_from_request(
        getattr(req_context, "kv_hints", None), getattr(req_context, "kv_transfer_params", None)
    )
    names = frozenset(host(h.endpoint) for h in kvhints.parse(env).fetches) if env is not None else frozenset()
    if state is not None:
        state[_Peers] = _Peers(names)
    return names


class FetchGate:
    """Wraps a RaidenOffloadingManager so peer reads need a fetch hint.

    A store is named by its job name. Deployments set ``raiden_job_name`` to
    the pod IP, so the host in a hint's endpoint names the store.
    """

    def __init__(self, manager: Any, mode: str = "hint") -> None:
        if mode not in MODES:
            raise ValueError(f"llmd_peer_read must be one of {MODES}, not {mode!r}")
        self._manager = manager
        self._mode = mode
        self._active: Any = _ANY  # peers allowed for the lookup in progress
        self._lookup = manager.lookup
        self._probe = manager._probe
        self.allowed = 0
        self.refused = 0
        manager.lookup = self.lookup
        manager._probe = self.probe

    def lookup(self, key, req_context):
        if self._mode == "always":
            return self._lookup(key, req_context)
        self._active = hinted_peers(req_context)
        try:
            return self._lookup(key, req_context)
        finally:
            self._active = _ANY

    def probe(self, sub_hashes, *, enable_global: bool = False):
        peers = self._active
        if not enable_global or peers is _ANY:
            return self._probe(sub_hashes, enable_global=enable_global)
        if not peers:
            # No hint: do not ask the registry at all.
            return self._probe(sub_hashes, enable_global=False)
        matches = self._probe(sub_hashes, enable_global=True)
        remote = self._manager._block_status.REMOTE
        for i, (_, block) in enumerate(matches):
            if block.status == remote:
                if block.raiden_id.job_name not in peers:
                    # Held by a peer the router did not name. The store
                    # answers with a found prefix, so cut the answer here.
                    self.refused += 1
                    return matches[:i]
                self.allowed += 1
        return matches
