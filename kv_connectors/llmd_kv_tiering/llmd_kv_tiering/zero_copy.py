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
"""Zero-copy peer loads into the engine's own CPU tier.

The engine registers its CPU-tier memory with NIXL. To load hinted blocks,
it asks its daemon to have the source daemon push them: the source pins
the blocks, and its RDMA helper writes them straight into the engine's
CPU-tier chunks. The blocks never pass through the local pool, so there
is no second copy. A push (RDMA WRITE) also runs at about twice the
bandwidth of a pull (RDMA READ) on 4 KiB-page memory.

If a push fails in a way that leaves its state unknown, writes into this
memory may still land after vLLM has reused it. So the reader then
registers the memory under a new NIXL agent and deregisters the old one.
That invalidates the keys the source holds, so a late write fails at the
NIC instead of landing.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Sequence

from llmd_kv_tiering.client import FETCH_MISSING, FETCH_OK, DaemonClient, DaemonError

logger = logging.getLogger("vllm.llmd_kv_tiering.zero_copy")


class ZeroCopyReader:
    """Lets peers write blocks into a registered local memory region."""

    def __init__(self, region_base: int, region_size: int, block_bytes: int, parallel: int, timeout_s: float):
        self._base = region_base
        self._size = region_size
        self._block = block_bytes
        self._name = f"vllm-{socket.gethostname()}-{os.getpid()}"
        self._generation = 0
        self._agent, self._reg, self._metadata = self._register()

    def _register(self):
        from nixl._api import nixl_agent, nixl_agent_config

        # A new name per registration, so sources load the new keys.
        name = f"{self._name}-{self._generation}"
        agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
        reg = agent.register_memory([(self._base, self._size, 0, "")], "DRAM")
        logger.info("zero-copy target %s: registered %d bytes at 0x%x", name, self._size, self._base)
        return agent, reg, agent.get_agent_metadata()

    def _reregister(self) -> None:
        """Invalidate the keys sources hold for this memory."""
        old_agent, old_reg = self._agent, self._reg
        self._generation += 1
        self._agent, self._reg, self._metadata = self._register()
        old_agent.deregister_memory(old_reg)
        logger.warning("zero-copy target re-registered after a push with unknown state")

    def read(
        self,
        client: DaemonClient,
        endpoint: str,
        keys: Sequence[bytes],
        chunk_ids: Sequence[int],
    ) -> tuple[list[bytes], list[bytes]]:
        """Have the source at endpoint write keys into the given chunks.

        Returns (loaded, missing). Keys in neither list failed and should take
        the pooled path. Raises DaemonError when the source cannot push, so
        the caller can fall back for all of them.
        """
        addrs = []
        for cid in chunk_ids:
            if not 0 <= cid * self._block < self._size:
                raise DaemonError(f"chunk {cid} outside the registered region")
            addrs.append(self._base + cid * self._block)
        try:
            resp = client.call(
                "remote_push",
                keys=list(keys),
                endpoint=endpoint,
                dst_meta=self._metadata,
                dst_addrs=addrs,
            )
        except DaemonError as e:
            # "no_rdma" means the source started nothing. Any other failure
            # may leave writes in flight into these chunks, which the caller
            # is about to fill another way.
            if str(e) != "no_rdma":
                self._reregister()
            raise
        loaded = [k for k, st in zip(keys, resp["status"]) if st == FETCH_OK]
        missing = [k for k, st in zip(keys, resp["status"]) if st == FETCH_MISSING]
        return loaded, missing
