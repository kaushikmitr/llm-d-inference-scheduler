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
"""kvtierd tier for vLLM on TPU (tpu-inference).

vLLM on TPU does not use vLLM's tiering framework. It uses tpu-inference's
TPUOffloadConnector, which keeps its own host-memory chunk store. This module
subclasses that connector; load it with no tpu-inference change:

    --kv-transfer-config '{
      "kv_connector": "TPUOffloadConnector",
      "kv_connector_module_path": "llmd_kv_tiering.tpu_connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "llmd_socket_path": "/run/kvtier/kvtierd.sock"
      }
    }'

How it works:

- Save. Every chunk the connector stores in host memory is also copied into
  the node's kvtierd pool, on a background thread.
- Lookup. The scheduler counts blocks in the pool as hits after the host-memory
  hits run out. It pins them, so they cannot be evicted before the load, and
  gives them virtual chunk ids above VIRTUAL_BASE.
- Load. The worker's chunk store returns the pool bytes of a virtual chunk as
  a numpy array. The connector's own host-to-device transfer and scatter then
  run unchanged. The scheduler unpins once the load is reported done.
- Fetch hints. A kv.fetch hint in kv_transfer_params["kv_hints"] makes the
  scheduler pull the request's blocks from the named daemon into the local
  pool first. A failed pull only means the blocks are recomputed.

Scope: single-host replicas. With several hosts, each host holds only its
shard of a chunk, and a block may count as a hit only when every host's
daemon holds it. That is not handled yet, so the connector refuses to start
on more than one host.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import logging
import os
import socket
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tpu_inference.offload.tpu_offload_connector import (
    TPUOffloadConnector as _BaseTPUOffloadConnector,
)

from llmd_kv_tiering import hints as kvhints
from llmd_kv_tiering.client import (
    ALLOCATED,
    FETCH_OK,
    FETCH_PRESENT,
    DaemonClient,
    DaemonError,
)

logger = logging.getLogger("vllm.llmd_kv_tiering.tpu")

DEFAULT_SOCKET = "/run/kvtier/kvtierd.sock"
# Chunk ids at or above this name pool slots, not host-memory chunks.
VIRTUAL_BASE = 1 << 40
_GROUP = b"\x00\x00\x00\x00"  # vLLM offload keys end in a 4-byte group index


def offload_key(block_hash: bytes) -> bytes:
    """The daemon key of a block: its hash plus group index 0, the same
    layout as vLLM's OffloadKey, so router hashes match on GPU and TPU."""
    return bytes(block_hash) + _GROUP


def tpu_namespace(vllm_config: Any) -> str:
    """Name the block layout. The scheduler and the worker compute the same
    value from the config alone."""
    mc, cc, pc = vllm_config.model_config, vllm_config.cache_config, vllm_config.parallel_config
    layout = {
        "engine": "tpu-inference",
        "model": mc.model,
        "dtype": str(mc.dtype),
        "kv_dtype": str(cc.cache_dtype),
        "block_size": cc.block_size,
        "tp": pc.tensor_parallel_size,
        "host": 0,
    }
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    return f"tpu:{mc.model}:{digest[:16]}"


def _extra(vllm_config: Any) -> dict:
    ktc = vllm_config.kv_transfer_config
    return dict(getattr(ktc, "kv_connector_extra_config", None) or {})


class LLMDTPUOffloadConnector(_BaseTPUOffloadConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = _extra(vllm_config)
        self._llmd_socket = extra.get("llmd_socket_path", DEFAULT_SOCKET)
        self._llmd_ns = extra.get("llmd_namespace") or tpu_namespace(vllm_config)
        self._llmd_timeout = float(extra.get("llmd_timeout_s", 30))
        self._llmd_sched: _SchedulerTier | None = None
        self._llmd_worker: _WorkerTier | None = None
        if self.connector_scheduler is not None:
            self._llmd_sched = _SchedulerTier(self, self.connector_scheduler)
        if self.connector_worker is not None:
            self._llmd_worker = _WorkerTier(self, self.connector_worker)

    # Scheduler side ----------------------------------------------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self._llmd_sched is not None:
            self._llmd_sched.before_match(request)
        try:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        finally:
            if self._llmd_sched is not None:
                self._llmd_sched.after_match()

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        if self._llmd_sched is not None:
            self._llmd_sched.annotate(meta)
        return meta

    def request_finished(self, request, block_ids):
        if self._llmd_sched is not None:
            self._llmd_sched.release_request(request.request_id)
        return super().request_finished(request, block_ids)

    # Worker side -------------------------------------------------------

    def register_runner(self, runner) -> None:
        super().register_runner(runner)
        if self._llmd_worker is not None:
            self._llmd_worker.on_runner()

    def start_load_kv(self, fwd_ctx, **kwargs) -> None:
        if self._llmd_worker is not None:
            self._llmd_worker.take_meta(self._get_connector_metadata())
        super().start_load_kv(fwd_ctx, **kwargs)

    def wait_for_save(self):
        if self._llmd_worker is not None:
            self._llmd_worker.take_meta(self._get_connector_metadata())
        super().wait_for_save()

    def shutdown(self):
        if self._llmd_sched is not None:
            self._llmd_sched.shutdown()
        if self._llmd_worker is not None:
            self._llmd_worker.shutdown()
        super().shutdown()


# The TPU platform accepts only a fixed list of connector names, and vLLM
# loads the class by name from kv_connector_module_path. Exporting the
# subclass under the base class's name passes that check with no patch.
TPUOffloadConnector = LLMDTPUOffloadConnector


# -------------------------------------------------------------------------
# Scheduler
# -------------------------------------------------------------------------


class _ManagerProxy:
    """Wraps the scheduler's LRUCacheManager to add pool hits."""

    def __init__(self, base, tier: _SchedulerTier) -> None:
        self._base = base
        self._tier = tier

    def __getattr__(self, name):
        return getattr(self._base, name)

    def lookup(self, chunk_hashes):
        n = self._base.lookup(chunk_hashes)
        return n + self._tier.count_pool_hits(chunk_hashes[n:])

    def prepare_load(self, chunk_hashes):
        chunks = []
        for h in chunk_hashes:
            c = self._base.cpu_cache.get(h)
            if c is not None and c.is_ready_to_load:
                chunks.extend(self._base.prepare_load([h]))
            else:
                chunks.append(self._tier.virtual_chunk(h))
        return chunks

    def mark_completion(self, chunk_ids, operation):
        real = [c for c in chunk_ids if c < VIRTUAL_BASE]
        virtual = [c for c in chunk_ids if c >= VIRTUAL_BASE]
        if virtual and operation == "load":
            self._tier.finish_virtual(virtual)
        if real:
            self._base.mark_completion(real, operation)


class _SchedulerTier:
    def __init__(self, conn: LLMDTPUOffloadConnector, sched) -> None:
        self._conn = conn
        self._sched = sched
        self._mirror: set[bytes] = set()
        self._client: DaemonClient | None = None
        self._lock = threading.Lock()
        self._next_virtual = VIRTUAL_BASE
        self._virtual: dict[int, tuple[bytes, int]] = {}  # id -> (key, slot)
        # New this step: id -> (slot, pool id). The pool id lets the worker
        # refuse a slot of a pool that a restarted daemon has replaced.
        self._step_virtual: dict[int, tuple[int, int]] = {}
        self._match_req = None
        # Pins taken while matching, per request, not yet turned into loads:
        # key -> (slot, pool id).
        self._held: dict[str, dict[bytes, tuple[int, int]]] = {}
        self._fetched_reqs: set[str] = set()
        # Events that arrive while a connection is set up, applied after its
        # snapshot. Only events change the mirror; they arrive in the order
        # the pool changed.
        self._connecting: list[dict] | None = None
        self._stop = threading.Event()
        sched.offload_manager = _ManagerProxy(sched.offload_manager, self)
        self._thread = threading.Thread(target=self._connect_loop, name="llmd-tpu-sched", daemon=True)
        self._thread.start()

    # connection

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        client = self._client
        if client is not None:
            client.close()

    def _connect_loop(self) -> None:
        while not self._stop.is_set():
            if self._client is None:
                self._connect()
            self._stop.wait(1.0)

    def _connect(self) -> None:
        holder: list[DaemonClient] = []
        with self._lock:
            self._connecting = []
        try:
            c = DaemonClient(
                self._conn._llmd_socket,
                self._conn._llmd_ns,
                0,  # attach: the worker creates the pool
                f"tpu-sched/{socket.gethostname()}/{os.getpid()}",
                on_event=self._on_event,
                on_disconnect=lambda: self._on_disconnect(holder[0] if holder else None),
                timeout_s=self._conn._llmd_timeout,
            )
            resp = c.call("snapshot")
        except (OSError, DaemonError) as e:
            logger.debug("scheduler cannot attach to kvtierd yet: %s", e)
            with self._lock:
                self._connecting = None
            return
        with self._lock:
            holder.append(c)
            buffered, self._connecting = self._connecting or [], None
            if not c.connected:
                return
            self._client = c
            self._mirror.update(resp.get("keys") or ())
            # The snapshot already reflects events received before its reply.
            for msg in buffered:
                if msg["_rx"] > resp["_rx"]:
                    self._apply_event(msg)
        logger.info("llm-d TPU tier: scheduler attached, %d blocks in pool", len(resp.get("keys") or ()))

    def _on_event(self, msg) -> None:
        with self._lock:
            if self._connecting is not None:
                self._connecting.append(msg)
                return
            self._apply_event(msg)

    def _apply_event(self, msg) -> None:
        self._mirror.update(msg.get("stored") or ())
        self._mirror.difference_update(msg.get("removed") or ())

    def _on_disconnect(self, client: DaemonClient | None) -> None:
        with self._lock:
            if client is None or self._client is not client:
                return
            self._client = None
            self._mirror.clear()

    # matching

    def before_match(self, request) -> None:
        self._match_req = request
        # Pins from an earlier match of this request that never became a
        # load are dropped; this match takes fresh ones.
        self.release_request(request.request_id, keep_fetch_mark=True)
        self._maybe_fetch(request)

    def after_match(self) -> None:
        self._match_req = None

    def _maybe_fetch(self, request) -> None:
        client = self._client
        if client is None or request.request_id in self._fetched_reqs:
            return
        params = getattr(request, "kv_transfer_params", None)
        env = kvhints.envelope_from_request(getattr(request, "kv_hints", None), params)
        if env is None:
            return
        self._fetched_reqs.add(request.request_id)
        parsed = kvhints.parse(env)
        fetches = [h for h in parsed.fetches if not h.evict_source]
        if not fetches:
            return
        hint = fetches[0]  # opportunistic: the whole request from one source
        base = self._sched.offload_manager._base
        hashes = self._sched._get_request_block_hashes(request)
        n = base.lookup(hashes)
        keys = [offload_key(h) for h in hashes[n:] if offload_key(h) not in self._mirror]
        if not keys:
            return
        try:
            status = client.call("fetch", keys=keys, endpoint=hint.endpoint, evict=False)["status"]
        except DaemonError as e:
            logger.warning("fetch of %d blocks from %s failed: %s", len(keys), hint.endpoint, e)
            return
        # The daemon's stored events, which arrive before this reply, add
        # the fetched blocks to the mirror.
        got = [k for k, st in zip(keys, status) if st in (FETCH_OK, FETCH_PRESENT)]
        logger.info(
            "request %s: pulled %d of %d blocks from %s",
            request.request_id,
            len(got),
            len(keys),
            hint.endpoint,
        )

    def count_pool_hits(self, chunk_hashes) -> int:
        """Leading blocks of chunk_hashes in the pool. Pins each one."""
        client, req = self._client, self._match_req
        if client is None or req is None:
            return 0
        keys = []
        for h in chunk_hashes:
            k = offload_key(h)
            if k not in self._mirror:
                break
            keys.append(k)
        if not keys:
            return 0
        try:
            slots = client.call("pin", keys=keys)["slots"]
        except DaemonError:
            return 0
        held = self._held.setdefault(req.request_id, {})
        n = 0
        for k, s in zip(keys, slots):
            if s < 0:  # evicted since the mirror saw it: hits stop here
                break
            held[k] = (s, client.pool_id)
            n += 1
        # Blocks after the gap cannot be used; release their pins at once.
        tail = [k for k, s in zip(keys[n + 1 :], slots[n + 1 :]) if s >= 0]
        if tail:
            self._unpin(tail)
        return n

    def virtual_chunk(self, block_hash):
        req = self._match_req
        k = offload_key(block_hash)
        held = self._held.get(req.request_id, {}) if req is not None else {}
        slot = held.pop(k, None)
        if slot is None:
            # update_state_after_alloc runs outside get_num_new_matched_tokens;
            # find the request that holds this key.
            for rid, h in self._held.items():
                if k in h:
                    slot = h.pop(k)
                    break
        if slot is None:
            raise RuntimeError(f"llm-d TPU tier: no pinned slot for block {k.hex()[:16]}")
        vid = self._next_virtual
        self._next_virtual += 1
        self._virtual[vid] = (k, slot[0])
        self._step_virtual[vid] = slot
        return types.SimpleNamespace(chunk_id=vid)

    def finish_virtual(self, vids) -> None:
        keys = [self._virtual.pop(v)[0] for v in vids if v in self._virtual]
        if keys:
            self._unpin(keys)

    def release_request(self, req_id: str, keep_fetch_mark: bool = False) -> None:
        held = self._held.pop(req_id, None)
        if held:
            self._unpin(list(held))
        if not keep_fetch_mark:
            self._fetched_reqs.discard(req_id)

    def _unpin(self, keys) -> None:
        client = self._client
        if client is None:
            return
        # The daemon drops a lost session's pins.
        with contextlib.suppress(DaemonError):
            client.call("unpin", keys=keys)

    def annotate(self, meta) -> None:
        meta.llmd_virtual = self._step_virtual
        self._step_virtual = {}
        id_to_hash = self._sched.offload_manager._base.chunk_pool.allocated_id_to_hash_map
        saves = {}
        for rm in getattr(meta, "requests_meta", ()):
            spec = getattr(rm, "save_spec", None)
            if spec is None or spec.skip_save:
                continue
            for cid in spec.dst_chunks:
                h = id_to_hash.get(cid)
                if h is not None:
                    saves[cid] = offload_key(h)
        meta.llmd_save = saves


# -------------------------------------------------------------------------
# Worker
# -------------------------------------------------------------------------


class _BackendProxy:
    """Wraps the worker's LocalCPUBackend: serves virtual chunks from the
    pool, and copies every stored chunk into the pool."""

    def __init__(self, base, tier: _WorkerTier) -> None:
        self._base = base
        self._tier = tier

    def __getattr__(self, name):
        return getattr(self._base, name)

    def get(self, chunk_id):
        if chunk_id >= VIRTUAL_BASE:
            return self._tier.read_virtual(chunk_id)
        return self._base.get(chunk_id)

    def add(self, chunk_id, value) -> bool:
        ok = self._base.add(chunk_id, value)
        self._tier.store(chunk_id, value)
        return ok


class _WorkerTier:
    def __init__(self, conn: LLMDTPUOffloadConnector, worker) -> None:
        self._conn = conn
        self._worker = worker
        self._client: DaemonClient | None = None
        self._virtual: dict[int, tuple[int, int]] = {}  # id -> (slot, pool id)
        self._save_keys: dict[int, bytes] = {}  # chunk id -> key
        self._shape: tuple[int, ...] | None = None
        self._dtype = None
        self._slot_bytes = 0
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llmd-tpu-save")
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._reconnector: threading.Thread | None = None
        worker.cpu_backend = _BackendProxy(worker.cpu_backend, self)

    def on_runner(self) -> None:
        import jax

        if jax.process_count() != 1:
            raise RuntimeError(
                "LLMDTPUOffloadConnector supports single-host replicas only "
                f"(this replica spans {jax.process_count()} hosts)"
            )
        w = self._worker
        if len(w.grouped_layer_indices) != 1:
            raise RuntimeError("LLMDTPUOffloadConnector supports one KV layer shape only")
        # A stacked chunk: [1, num_layers, block_size, ...] in the cache dtype.
        self._shape = (1, w.num_layers) + tuple(w.shape[1:])
        self._dtype = np.dtype(w.dtype)
        self._slot_bytes = int(np.prod(self._shape)) * self._dtype.itemsize
        if not self._connect():
            # The worker creates the pool, so it must reach the daemon once.
            raise RuntimeError(f"llm-d TPU tier: cannot reach kvtierd at {self._conn._llmd_socket}")
        self._reconnector = threading.Thread(target=self._reconnect_loop, name="llmd-tpu-worker", daemon=True)
        self._reconnector.start()

    def _connect(self) -> bool:
        holder: list[DaemonClient] = []
        try:
            client = DaemonClient(
                self._conn._llmd_socket,
                self._conn._llmd_ns,
                self._slot_bytes,
                f"tpu-worker/{socket.gethostname()}/{os.getpid()}",
                on_event=lambda m: None,
                on_disconnect=lambda: self._on_disconnect(holder[0] if holder else None),
                timeout_s=self._conn._llmd_timeout,
            )
        except (OSError, DaemonError) as e:
            logger.debug("llm-d TPU tier: worker cannot reach kvtierd: %s", e)
            return False
        holder.append(client)
        if not client.connected:
            return False
        self._client = client
        logger.info(
            "llm-d TPU tier: worker connected, chunk shape %s %s, %d bytes, ns %s",
            self._shape,
            self._dtype,
            self._slot_bytes,
            self._conn._llmd_ns,
        )
        return True

    def _on_disconnect(self, client: DaemonClient | None) -> None:
        if client is None or self._client is not client:
            return
        logger.warning("llm-d TPU tier: worker lost kvtierd; saves to the pool pause until it is back")
        self._client = None
        self._lost.set()

    def _reconnect_loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            self._lost.wait()
            if self._stop.is_set():
                return
            if self._client is not None:
                self._lost.clear()
                continue
            if self._connect():
                self._lost.clear()
                backoff = 0.5
                continue
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 10.0)

    def shutdown(self) -> None:
        self._stop.set()
        self._lost.set()
        if self._reconnector is not None:
            self._reconnector.join(timeout=5)
        self._pool.shutdown(wait=True)
        client = self._client
        if client is not None:
            client.close()

    def take_meta(self, meta) -> None:
        v = getattr(meta, "llmd_virtual", None)
        if v:
            self._virtual.update(v)
        s = getattr(meta, "llmd_save", None)
        if s:
            self._save_keys.update(s)

    def read_virtual(self, vid: int):
        client = self._client
        entry = self._virtual.pop(vid, None)
        if client is None or entry is None:
            logger.error("llm-d TPU tier: virtual chunk %d unavailable", vid)
            return None
        slot, pool_id = entry
        if pool_id != client.pool_id:
            # The daemon restarted between the scheduler's pin and this
            # load. The slot number belongs to the old pool.
            logger.error("llm-d TPU tier: virtual chunk %d is from a replaced pool", vid)
            return None
        addr = client.arena.slot_address(slot)
        buf = (ctypes.c_char * self._slot_bytes).from_address(addr)
        # The slot stays pinned by the scheduler until this load is reported
        # done, so a view is safe; device_put copies it to the TPU.
        return np.frombuffer(buf, dtype=self._dtype).reshape(self._shape)

    def store(self, chunk_id: int, value) -> None:
        key = self._save_keys.pop(chunk_id, None)
        if key is None or self._client is None or isinstance(value, list):
            return
        self._pool.submit(self._store, key, value)

    def _store(self, key: bytes, value) -> None:
        client = self._client
        if client is None:
            return
        try:
            arr = np.asarray(value)
            if arr.nbytes != self._slot_bytes:
                logger.warning(
                    "llm-d TPU tier: chunk of %d bytes, expected %d; not stored",
                    arr.nbytes,
                    self._slot_bytes,
                )
                return
            arr = np.ascontiguousarray(arr)
            resp = client.call("alloc", keys=[key])
            if resp["status"][0] != ALLOCATED:
                return
            slot = resp["slots"][0]
            ctypes.memmove(client.arena.slot_address(slot), arr.ctypes.data, self._slot_bytes)
            client.call("commit", keys=[key], ok=[True])
        except Exception:  # noqa: BLE001 - a failed copy only loses a pool entry
            logger.exception("llm-d TPU tier: storing a chunk in the pool failed")
