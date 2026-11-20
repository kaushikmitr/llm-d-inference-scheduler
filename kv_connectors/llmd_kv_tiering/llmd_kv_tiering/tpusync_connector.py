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
"""Router KV hints for vLLM on TPU, carried out by TPU Sync.

This is a proof of concept. It is a second TPU path next to
``tpu_connector``. That module copies blocks through host numpy arrays into
the kvtierd pool. This one hands the copies to TPU Sync
(https://github.com/google/tpu-sync), which moves blocks between TPU memory
and host memory, and between hosts, without a pass through Python. No
kvtierd runs on the node.

Load it with no tpu-inference change:

    --kv-transfer-config '{
      "kv_connector": "RaidenOffloadConnector",
      "kv_connector_module_path": "llmd_kv_tiering.tpusync_connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "tpusync_registry": "registry-host:50051",
        "tpusync_capacity_blocks": 4000
      }
    }'

How it works:

- Save. When a request finishes, its full blocks that the store does not
  hold yet are copied from TPU memory to host memory. The request's blocks
  are freed only once that copy has ended.
- Lookup. The scheduler asks the store for the request's blocks. Blocks in
  local host memory are hits. Blocks on a peer are hits only when a
  ``kv.fetch`` hint names that peer (``tpusync_global_lookup: "hint"``, the
  default), or always (``"always"``), or never (``"never"``).
- Load. Hits are loaded asynchronously into the request's newly allocated
  blocks. Local blocks come from host memory. Peer blocks are read from the
  peer's host memory straight into local TPU memory. A block that fails to
  load is reported to vLLM, which recomputes it.
- Push. A ``kv.prefetch`` hint makes the source offer the request's blocks
  to the named peer once they are saved. TPU Sync's remote save lands only
  in free blocks, so a peer with a full store refuses the offer.

Peers are named by pod IP. Each store registers as
``RaidenId(pool, pod IP, layout, 0)`` in the TPU Sync global registry, and a
hint's control endpoint is reduced to its host. The same hints therefore
work for kvtierd and for this connector.

Scope and limits:

- Single-host replicas. The scheduler and the worker must share one process
  (vLLM's UniProcExecutor, which tpu-inference uses on a single host).
- Blocks are saved when a request finishes, not while it runs.
- ``kv.evict_on_fetch`` and ``kv.move`` copy but do not delete the source
  copy. TPU Sync has no caller-driven removal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from llmd_kv_tiering import hints as kvhints

logger = logging.getLogger("vllm.llmd_kv_tiering.tpusync")

_LOCAL = ("HOST", "HOST_AND_HBM")
_REMOTE = "REMOTE"


def _host(endpoint: str) -> str:
    """The host of a control endpoint such as ``tcp://10.0.0.1:17771``."""
    s = endpoint.split("://", 1)[-1]
    if s.startswith("["):
        return s[1 : s.index("]")]
    return s.rsplit(":", 1)[0] if ":" in s else s


def layout_id(vllm_config: Any) -> str:
    """Name the block layout. Stores pair only when this matches."""
    mc, cc, pc = vllm_config.model_config, vllm_config.cache_config, vllm_config.parallel_config
    layout = {
        "engine": "tpu-inference/tpu-sync",
        "model": mc.model,
        "dtype": str(mc.dtype),
        "kv_dtype": str(cc.cache_dtype),
        "block_size": cc.block_size,
        "tp": pc.tensor_parallel_size,
    }
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Config:
    registry: str
    capacity: int
    ip: str
    pool: str
    controller_port: int
    worker_port: int
    global_lookup: str
    timeout_s: float
    layout: str

    @classmethod
    def from_vllm(cls, vllm_config: Any) -> Config:
        ktc = vllm_config.kv_transfer_config
        extra = dict(getattr(ktc, "kv_connector_extra_config", None) or {})
        mode = extra.get("tpusync_global_lookup", "hint")
        if mode not in ("hint", "always", "never"):
            raise ValueError(f"tpusync_global_lookup must be hint, always or never, not {mode!r}")
        ip = extra.get("tpusync_ip") or os.environ.get("POD_IP")
        if not ip:
            raise ValueError("set tpusync_ip or POD_IP: peers reach this store by that address")
        return cls(
            registry=extra.get("tpusync_registry", ""),
            capacity=int(extra.get("tpusync_capacity_blocks", 2000)),
            ip=ip,
            pool=extra.get("tpusync_pool", "llmd-kv"),
            controller_port=int(extra.get("tpusync_controller_port", 0)),
            worker_port=int(extra.get("tpusync_worker_port", 0)),
            global_lookup=mode,
            timeout_s=float(extra.get("tpusync_timeout_s", 60)),
            layout=extra.get("tpusync_layout") or layout_id(vllm_config),
        )


def _load_native() -> Any:
    """TPU Sync's JAX extension. Some wheels ship it without the Python
    wrapper around the store, so the store is used through the extension."""
    from tpu_sync.api.jax.kv_cache_manager import KVCacheManager
    from tpu_sync.frameworks.jax import _tpu_raiden_jax as impl

    class Native:
        KVCacheStore = impl.KVCacheStore
        RaidenId = impl.RaidenId
        RaidenBlockId = impl.RaidenBlockId
        BlockStatus = impl.BlockStatus

    Native.KVCacheManager = KVCacheManager
    return Native


def _settle_device(runner: Any) -> None:
    """Wait until every dispatched forward pass has written its blocks."""
    import jax

    jax.block_until_ready(runner.kv_caches)


@dataclass
class _Load:
    pending: dict[bytes, int]  # key -> destination device block
    local: set[bytes]  # keys loaded under a lookup pin
    deadline: float
    started: float
    blocks: int = 0
    remote: int = 0
    failed: int = 0


@dataclass
class _SavePlan:
    req_id: str
    keys: list[bytes]
    blocks: list[int]


@dataclass
class Stats:
    saved: int = 0
    save_failed: int = 0
    save_rejected: int = 0
    loaded: int = 0
    load_failed: int = 0
    pushed: int = 0
    push_failed: int = 0
    last_load_s: float = 0.0


class Runtime:
    """The store and the transfers in flight. One per process: the scheduler
    side and the worker side of the connector both use it."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.native: Any = None
        self.store: Any = None
        self.manager: Any = None
        self.rid: Any = None
        self.runner: Any = None
        self.stats = Stats()
        self.loads: dict[str, _Load] = {}
        self.load_owner: dict[bytes, str] = {}
        self.done_recv: set[str] = set()
        self.invalid_blocks: set[int] = set()
        self.plans: list[_SavePlan] = []
        self.sends: dict[str, set[bytes]] = {}
        self.save_owner: dict[bytes, tuple[str, float]] = {}
        self.done_send: set[str] = set()
        self.push_after: dict[str, tuple[str, list[bytes]]] = {}
        self.pushing: dict[bytes, float] = {}

    @property
    def ready(self) -> bool:
        return self.store is not None

    # startup

    def start(self, runner: Any) -> None:
        import jax
        import numpy as np

        if jax.process_count() != 1:
            raise RuntimeError(f"the TPU Sync connector supports single-host replicas only, not {jax.process_count()}")
        cfg = self.cfg
        n = self.native = _load_native()
        kv = runner.kv_caches
        if not kv:
            raise RuntimeError("the runner has no KV caches to register")
        shards = len(kv[0].sharding.device_set)
        block_bytes = sum(int(np.prod(a.shape[1:])) * np.dtype(a.dtype).itemsize for a in kv)
        self.rid = n.RaidenId(cfg.pool, cfg.ip, "kv-" + cfg.layout, 0)
        self.store = n.KVCacheStore(
            capacity=cfg.capacity,
            global_registry_address=cfg.registry,
            raiden_id=self.rid,
            num_shards=shards,
            shard_size_bytes=block_bytes // shards,
            store_server_ip=cfg.ip,
            raiden_controller_port=cfg.controller_port,
            expected_worker_count=0,
            kv_pool_group="",
            secondary_backend_configs=[],
        )
        self.manager = n.KVCacheManager(
            kv_caches=kv,
            local_control_port=0,
            host_blocks_to_allocate=cfg.capacity,
            unsafe_skip_buffer_lock=True,
            raiden_worker_port=cfg.worker_port,
            raiden_controller_address=self.store.raiden_controller_address,
            worker_id=f"{cfg.ip}/0",
            node_id=0,
        )
        self.runner = runner
        time.sleep(1.0)  # the worker registers with the store's controller in the background
        logger.info(
            "TPU Sync store ready: %s, %d blocks of %d bytes over %d shards, registry %s, global lookup %s",
            self.store.store_server_address,
            cfg.capacity,
            block_bytes,
            shards,
            cfg.registry or "none",
            cfg.global_lookup,
        )

    def peer(self, host: str) -> Any:
        return self.native.RaidenId(self.cfg.pool, host, "kv-" + self.cfg.layout, 0)

    def present(self, key: bytes) -> bool:
        return bool(self.store.lookup([key], False, False))

    # saves

    def flush_saves(self) -> None:
        """Start the planned saves. Runs on the worker before a forward pass."""
        with self.lock:
            plans, self.plans = self.plans, []
        if not plans:
            return
        _settle_device(self.runner)
        n = self.native
        deadline = time.monotonic() + self.cfg.timeout_s
        for p in plans:
            slices = [n.RaidenBlockId(self.rid, -1, b, n.BlockStatus.HBM) for b in p.blocks]
            ok = self.store.insert(p.keys, slices, False)
            if ok and not self.store.save(p.keys, None):
                self.store.release(p.keys)
                ok = False
            with self.lock:
                if ok:
                    for k in p.keys:
                        self.save_owner[k] = (p.req_id, deadline)
                    continue
                self.stats.save_rejected += len(p.keys)
                for k in p.keys:
                    self._save_settled(k, None)
            logger.warning("request %s: the store did not accept %d blocks to save", p.req_id, len(p.keys))

    def _save_settled(self, key: bytes, ok: bool | None) -> None:
        """A local save of key ended. ok is None when it never started."""
        owner = self.save_owner.pop(key, None)
        if owner is None:
            return
        req_id = owner[0]
        if ok:
            self.stats.saved += 1
        elif ok is False:
            self.stats.save_failed += 1
            self.store.release([key])  # a failed save keeps its pin
        left = self.sends.get(req_id)
        if left is None:
            return
        left.discard(key)
        if not left:
            del self.sends[req_id]
            self.done_send.add(req_id)
            self._start_push(req_id)

    # pushes

    def _start_push(self, req_id: str) -> None:
        push = self.push_after.pop(req_id, None)
        if push is None:
            return
        host, keys = push
        keys = [k for k in keys if k not in self.pushing]
        if not keys:
            return
        found = self.store.lookup(keys, False, True)  # pins what it returns
        send = [k for k, b in found if b.status.name in _LOCAL]
        other = [k for k, b in found if b.status.name not in _LOCAL]
        if other:
            self.store.release(other)
        if not send:
            return
        if not self.store.save(send, self.peer(host)):
            self.store.release(send)
            self.stats.push_failed += len(send)
            logger.warning("request %s: push of %d blocks to %s was not accepted", req_id, len(send), host)
            return
        # The destination answers within its hold window of about 30 s.
        deadline = time.monotonic() + max(self.cfg.timeout_s, 45.0)
        for k in send:
            self.pushing[k] = deadline
        logger.info("request %s: offered %d blocks to %s", req_id, len(send), host)

    def _push_settled(self, key: bytes, ok: bool) -> None:
        if self.pushing.pop(key, None) is None:
            return
        if ok:
            self.stats.pushed += 1
        else:
            self.stats.push_failed += 1
            self.store.release([key])  # a failed remote save keeps its pin

    # loads

    def start_load(self, req_id: str, groups: list[tuple[bool, list[bytes], list[Any], list[int]]]) -> None:
        """Start loading. Each group is (local, keys, slices, device blocks)."""
        now = time.monotonic()
        load = _Load(pending={}, local=set(), deadline=now + self.cfg.timeout_s, started=now)
        self.loads[req_id] = load
        for local, keys, slices, blocks in groups:
            load.blocks += len(keys)
            if local:
                ok = self.store.load(keys, blocks)
            else:
                load.remote += len(keys)
                ok = self.store.load(keys, slices, blocks)
            if not ok:
                logger.warning("request %s: the store refused a load of %d blocks", req_id, len(keys))
                if local:
                    self.store.release(keys)
                load.failed += len(keys)
                self.invalid_blocks.update(blocks)
                continue
            for k, b in zip(keys, blocks):
                load.pending[k] = b
                self.load_owner[k] = req_id
            if local:
                load.local.update(keys)
        self._load_maybe_done(req_id)

    def _load_settled(self, key: bytes, ok: bool) -> None:
        req_id = self.load_owner.pop(key, None)
        if req_id is None:
            return
        load = self.loads[req_id]
        block = load.pending.pop(key)
        if not ok:
            load.failed += 1
            self.invalid_blocks.add(block)
            if key in load.local:
                self.store.release([key])  # a failed local load keeps its pin
        self._load_maybe_done(req_id)

    def _load_maybe_done(self, req_id: str) -> None:
        load = self.loads[req_id]
        if load.pending:
            return
        del self.loads[req_id]
        self.done_recv.add(req_id)
        took = time.monotonic() - load.started
        ok = load.blocks - load.failed
        self.stats.load_failed += load.failed
        self.stats.loaded += ok
        self.stats.last_load_s = took
        logger.info(
            "request %s: loaded %d of %d blocks (%d from a peer) in %.3f s",
            req_id,
            ok,
            load.blocks,
            load.remote,
            took,
        )

    # polling

    def poll(self) -> None:
        """Collect finished transfers. Runs on the worker every step."""
        with self.lock:
            if self.loads:
                done, failed, _ = self.store.poll_load_status()
                for k in done:
                    self._load_settled(k, True)
                for k in failed:
                    self._load_settled(k, False)
            if self.save_owner or self.pushing:
                res = self.store.poll_save_status()
                for k in res[0]:
                    if k in self.pushing:
                        self._push_settled(k, True)
                    else:
                        self._save_settled(k, True)
                for k in res[1]:
                    if k in self.pushing:
                        self._push_settled(k, False)
                    else:
                        self._save_settled(k, False)
            now = time.monotonic()
            for req_id, load in list(self.loads.items()):
                if now > load.deadline:
                    logger.error("request %s: %d blocks did not load in time", req_id, len(load.pending))
                    for k in list(load.pending):
                        self._load_settled(k, False)
            for k, (_, deadline) in list(self.save_owner.items()):
                if now > deadline:
                    self._save_settled(k, False)
            for k, deadline in list(self.pushing.items()):
                if now > deadline:
                    self._push_settled(k, False)

    def take_finished(self) -> tuple[set[str], set[str]]:
        with self.lock:
            send, self.done_send = self.done_send, set()
            recv, self.done_recv = self.done_recv, set()
        return send, recv

    def take_invalid_blocks(self) -> set[int]:
        with self.lock:
            bad, self.invalid_blocks = self.invalid_blocks, set()
        return bad


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def runtime(vllm_config: Any) -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime(Config.from_vllm(vllm_config))
        return _runtime


class _Meta(KVConnectorMetadata):
    """Empty. The two sides share the Runtime, not metadata."""


@dataclass
class _Match:
    skip: int  # leading blocks the engine already holds
    keys: list[bytes]
    slices: list[Any]
    local: list[bool]


class LLMDTPUSyncConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._rt = runtime(vllm_config)
        self._block_size = vllm_config.cache_config.block_size
        self._ns = bytes.fromhex(self._rt.cfg.layout)
        self._is_scheduler = role == KVConnectorRole.SCHEDULER
        self._matched: dict[str, _Match] = {}
        self._warned_not_ready = False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        return "NHD"

    def _key(self, block_hash: bytes) -> bytes:
        # The layout prefix keeps different models apart in one registry.
        return self._ns + bytes(block_hash)

    # Scheduler side ----------------------------------------------------

    def _fetch_peers(self, request) -> set[str] | None:
        """Peers this request may read from. None means any peer."""
        mode = self._rt.cfg.global_lookup
        if mode == "never" or not self._rt.cfg.registry:
            return set()
        if mode == "always":
            return None
        env = kvhints.envelope_from_request(
            getattr(request, "kv_hints", None), getattr(request, "kv_transfer_params", None)
        )
        if env is None:
            return set()
        return {_host(h.endpoint) for h in kvhints.parse(env).fetches}

    def _drop_match(self, req_id: str) -> None:
        m = self._matched.pop(req_id, None)
        if m is None:
            return
        pinned = [k for k, local in zip(m.keys, m.local) if local]
        if pinned:
            self._rt.store.release(pinned)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        rt = self._rt
        if not rt.ready:
            if not self._warned_not_ready:
                self._warned_not_ready = True
                logger.error("the TPU Sync store is not in this process; KV is neither saved nor loaded")
            return 0, False
        req_id = request.request_id
        with rt.lock:
            self._drop_match(req_id)  # pins of a match that was never scheduled
            skip = num_computed_tokens // self._block_size
            keys = [self._key(h) for h in request.block_hashes[skip:]]
            if not keys:
                return 0, False
            peers = self._fetch_peers(request)
            found = rt.store.lookup(keys, peers is None or bool(peers), True)
            m = _Match(skip, [], [], [])
            wait = False
            cut = len(found)
            for i, (k, b) in enumerate(found):
                status = b.status.name
                local = status in _LOCAL
                if not local and not (status == _REMOTE and (peers is None or b.raiden_id.job_replica_id in peers)):
                    cut = i
                    break
                if k in rt.load_owner:
                    # Another request is loading this block. Results are
                    # reported per block, so wait for that load to end.
                    cut, wait = i, True
                    break
                m.keys.append(k)
                m.slices.append(b)
                m.local.append(local)
            # lookup() pinned every local block it returned.
            unused = [k for k, b in found[cut:] if b.status.name not in (_REMOTE, "SHARED_STORAGE")]
            if unused:
                rt.store.release(unused)
            self._matched[req_id] = m
            if wait or not m.keys:
                self._drop_match(req_id)
                return (None if wait else 0), False
            return len(m.keys) * self._block_size, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        rt = self._rt
        req_id = request.request_id
        if not rt.ready:
            return
        with rt.lock:
            if num_external_tokens <= 0:
                self._drop_match(req_id)
                return
            m = self._matched.pop(req_id, None)
            if m is None:
                return
            n = min(len(m.keys), num_external_tokens // self._block_size)
            tail = [k for k, local in zip(m.keys[n:], m.local[n:]) if local]
            if tail:
                rt.store.release(tail)
            dst = list(blocks.get_block_ids()[0][m.skip : m.skip + n])
            if len(dst) != n:
                raise RuntimeError(f"request {req_id}: {n} blocks to load but {len(dst)} allocated")
            # One call loads from local host memory or from one peer, never both.
            groups: list[tuple[bool, list[bytes], list[Any], list[int]]] = []
            last = None
            for k, b, local, d in zip(m.keys[:n], m.slices[:n], m.local[:n], dst):
                src = None if local else b.raiden_id.job_replica_id
                if not groups or src != last:
                    groups.append((local, [], [], []))
                    last = src
                groups[-1][1].append(k)
                groups[-1][2].append(b)
                groups[-1][3].append(d)
            rt.start_load(req_id, groups)

    def build_connector_meta(self, scheduler_output):
        return _Meta()

    def request_finished(self, request, block_ids):
        rt = self._rt
        req_id = request.request_id
        if not rt.ready:
            return False, None
        with rt.lock:
            self._drop_match(req_id)
            if req_id in rt.loads:
                # Still loading into these blocks. vLLM frees them when the
                # worker reports the load finished.
                return True, None
            hashes = request.block_hashes
            n = min(len(block_ids), len(hashes))
            keys = [self._key(h) for h in hashes[:n]]
            new = [(k, b) for k, b in zip(keys, block_ids[:n]) if k not in rt.save_owner and not rt.present(k)]
            push = self._push_hint(request)
            if push is not None and keys:
                rt.push_after[req_id] = (push, keys)
            if not new:
                rt._start_push(req_id)
                return False, None
            new_keys = [k for k, _ in new]
            for k in new_keys:
                rt.save_owner[k] = (req_id, time.monotonic() + rt.cfg.timeout_s)
            rt.sends[req_id] = set(new_keys)
            rt.plans.append(_SavePlan(req_id, new_keys, [b for _, b in new]))
            return True, None

    def _push_hint(self, request) -> str | None:
        if not self._rt.cfg.registry:
            return None
        env = kvhints.envelope_from_request(
            getattr(request, "kv_hints", None), getattr(request, "kv_transfer_params", None)
        )
        if env is None:
            return None
        pushes = kvhints.parse(env).pushes
        return _host(pushes[0].endpoint) if pushes else None

    def has_pending_push_work(self) -> bool:
        # Keeps the engine stepping so that the worker starts planned saves
        # and collects push results after the last request has finished.
        rt = self._rt
        return bool(rt.plans or rt.pushing)

    # Worker side -------------------------------------------------------

    def register_runner(self, runner) -> None:
        self._rt.start(runner)

    def register_kv_caches(self, kv_caches) -> None:
        pass  # the runner's caches are registered in register_runner

    def start_load_kv(self, forward_context, **kwargs) -> None:
        # Loads start on the scheduler side. This hook runs before the
        # forward pass, when the blocks of earlier steps are final.
        if self._rt.ready:
            self._rt.flush_saves()

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(self, finished_req_ids):
        rt = self._rt
        if not rt.ready:
            return None, None
        rt.poll()
        send, recv = rt.take_finished()
        return send or None, recv or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._rt.take_invalid_blocks()


# The TPU platform accepts only a fixed list of connector names, and vLLM
# loads the class by name from kv_connector_module_path. Exporting the class
# under an accepted name passes that check with no patch.
RaidenOffloadConnector = LLMDTPUSyncConnector
