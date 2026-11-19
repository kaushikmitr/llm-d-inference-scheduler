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
"""Router KV hints.

An envelope is ``{protocol_version, message_id, actions: [...]}``; each
action is ``{action_id, action_type, action_version, payload}``. This is
vLLM's ``KvHintsEnvelope``. An envelope reaches the tier either as
``req_context.kv_hints`` (engines driven in process, such as Dynamo) or as
``kv_transfer_params["kv_hints"]`` (the OpenAI HTTP server, which llm-d's
router uses), so no engine change is needed.

Supported actions:

=====================  =============================================  ==========
action_type            payload                                        sent to
=====================  =============================================  ==========
kv.fetch               source_control_endpoint, block_hashes          destination
kv.evict_on_fetch      source_control_endpoint, block_hashes          destination
kv.prefetch            destination_control_endpoint, block_hashes     source
kv.move                destination_control_endpoint, block_hashes     source
=====================  =============================================  ==========

The action types, versions and payloads match those of vLLM's in-tree ``kvcr``
secondary tier, so a router can address either tier with the same hints.

Each action is parsed on its own; a bad action is rejected without
affecting the others. llm-d-specific actions will use an ``llmd.`` prefix.

Versions must match exactly. An envelope with another protocol_version is
rejected whole, and an action with another action_version is rejected,
because a different version may change what the fields mean.

How the tier applies fetch hints (``vllm_tier``):

- Precedence. When several fetch hints name the same block hash, the first
  one in the envelope wins.
- Opportunistic query. A plain ``kv.fetch`` covers every block of its
  request, not only the hashes it lists: any block of the request that is
  not local is fetched from that hint's source. Routers rely on this
  because 64-bit hashes do not survive every hint path (vLLM's gRPC hint
  path carries numbers as doubles), so llm-d's router sends a placeholder
  hash. Only the first plain ``kv.fetch`` is used this way. Blocks the
  source does not have are reported missing and recomputed. A
  ``kv.evict_on_fetch`` never applies beyond the hashes it names, since it
  deletes the source copy. Set ``opportunistic_query: false`` to fetch
  only listed hashes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "0.1"
ACTION_VERSION = "1.0"
TRANSFER_PARAMS_KEY = "kv_hints"

KV_FETCH = "kv.fetch"
KV_EVICT_ON_FETCH = "kv.evict_on_fetch"
KV_PREFETCH = "kv.prefetch"
KV_MOVE = "kv.move"

_PEER_FIELD = {
    KV_FETCH: "source",
    KV_EVICT_ON_FETCH: "source",
    KV_PREFETCH: "destination",
    KV_MOVE: "destination",
}

_MASK64 = (1 << 64) - 1
_warned: set[str] = set()


def _warn_once(issue: str, msg: str, *args: Any) -> None:
    if issue not in _warned:
        _warned.add(issue)
        logger.warning(msg, *args)


@dataclass(frozen=True)
class Hint:
    action_id: str
    action_type: str
    endpoint: str  # the source for fetches, the destination for pushes
    block_hashes: frozenset[int]

    @property
    def is_fetch(self) -> bool:
        return self.action_type in (KV_FETCH, KV_EVICT_ON_FETCH)

    @property
    def evict_source(self) -> bool:
        return self.action_type in (KV_EVICT_ON_FETCH, KV_MOVE)


@dataclass(frozen=True)
class Rejection:
    action_id: str | None
    action_type: str
    reason: str


@dataclass
class ParsedHints:
    hints: list[Hint] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def fetches(self) -> list[Hint]:
        return [h for h in self.hints if h.is_fetch]

    @property
    def pushes(self) -> list[Hint]:
        return [h for h in self.hints if not h.is_fetch]


def router_hash(offload_key: bytes) -> int:
    """The router's 64-bit hash of a vLLM offload key: the low 64 bits of
    its block hash (the key minus a 4-byte group index). Matches vLLM's
    maybe_convert_block_hash."""
    return int.from_bytes(offload_key[:-4], "big") & _MASK64


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def envelope_from_request(kv_hints: Any, kv_transfer_params: Mapping | None) -> Any:
    """The request's hint envelope, from either carrier, or None."""
    if kv_hints is not None:
        return kv_hints
    if kv_transfer_params:
        return kv_transfer_params.get(TRANSFER_PARAMS_KEY)
    return None


def _parse_action(raw: Any) -> Hint | Rejection:
    action_id = _get(raw, "action_id")
    if not isinstance(action_id, str):
        return Rejection(None, "unknown", "invalid_hint")
    action_type = _get(raw, "action_type")
    if not isinstance(action_type, str):
        return Rejection(action_id, "unknown", "invalid_hint")
    peer = _PEER_FIELD.get(action_type)
    if peer is None:
        return Rejection(action_id, action_type, "unsupported_action_type")
    version = _get(raw, "action_version")
    if version != ACTION_VERSION:
        reason = f"unsupported action_version {version!r}; supported {ACTION_VERSION!r}"
        return Rejection(action_id, action_type, reason)
    payload = _get(raw, "payload")
    if not isinstance(payload, Mapping):
        return Rejection(action_id, action_type, "invalid_payload")
    if "no_retain" in payload:
        return Rejection(action_id, action_type, "no_retain is not currently supported")
    endpoint = payload.get(f"{peer}_control_endpoint")
    if endpoint is None:
        return Rejection(action_id, action_type, f"{peer}-less hints are not currently supported")
    hashes = payload.get("block_hashes")
    if not isinstance(endpoint, str) or not endpoint or not isinstance(hashes, list):
        return Rejection(action_id, action_type, "invalid router hint")
    out: set[int] = set()
    for h in hashes:
        if isinstance(h, bool) or not isinstance(h, int) or not 0 <= h < 1 << 64:
            return Rejection(action_id, action_type, "invalid router hint")
        out.add(h)
    if not out:
        return Rejection(action_id, action_type, "invalid router hint")
    return Hint(action_id, action_type, endpoint, frozenset(out))


def parse(envelope: Any) -> ParsedHints:
    """Parse an envelope. Bad actions become rejections. A malformed
    envelope yields one rejection and no hints."""
    result = ParsedHints()
    if envelope is None:
        return result
    version = _get(envelope, "protocol_version")
    if version != PROTOCOL_VERSION:
        reason = f"unsupported protocol_version {version!r}; supported {PROTOCOL_VERSION!r}"
        _warn_once(f"rejected:envelope:{reason}", "KV hint envelope rejected; %s", reason)
        result.rejected.append(Rejection(None, "envelope", reason))
        return result
    actions = _get(envelope, "actions")
    if not isinstance(actions, list):
        result.rejected.append(Rejection(None, "envelope", "invalid router hint"))
        return result
    for raw in actions:
        try:
            parsed = _parse_action(raw)
        except Exception:  # noqa: BLE001 - one bad action must not sink the rest
            parsed = Rejection(None, "unknown", "invalid_hint")
        if isinstance(parsed, Hint):
            result.hints.append(parsed)
        else:
            _warn_once(
                f"rejected:{parsed.action_type}:{parsed.reason}",
                "KV hint rejected; action_type=%s reason=%s",
                parsed.action_type,
                parsed.reason,
            )
            result.rejected.append(parsed)
    return result
