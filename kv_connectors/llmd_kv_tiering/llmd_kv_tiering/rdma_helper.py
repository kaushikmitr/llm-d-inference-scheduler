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
"""RDMA helper for kvtierd.

Runs next to the daemon, one per node. It attaches to the daemon's pool,
maps it, and registers it with NIXL. The daemon then asks it, over a Unix
socket, to read blocks from a peer's pool straight into local pool slots
with RDMA. Peers exchange NIXL metadata through the daemons' own peer
protocol, so the helper opens no network port of its own.

Requests and responses are JSON lines:

    {"op": "info"}
        -> {"name", "metadata" (base64), "base", "slot_bytes"}
    {"op": "read", "peer", "peer_metadata" (base64), "remote_base",
     "slot_bytes", "pairs": [[local_slot, remote_slot], ...]}
        -> {} on success, {"error": "..."} on failure
    {"op": "write", "peer_metadata" (base64), "slot_bytes",
     "pairs": [[local_slot, remote_address], ...]}
        -> {} once every block is written to the peer's memory

RDMA WRITE is preferred where the source can push: on 4 KiB pages a
WRITE reaches about twice the bandwidth of a READ.

Transfer lifetime. A request may carry "timeout_s", counted from when the
helper receives it. Every transfer the helper starts reaches DONE or ERR
before it answers, so a plain error means no transfer is still running.
If the deadline passes with transfers still running, or their state
cannot be read, the helper answers {"error", "unsafe": true} and exits.
Exiting closes its RDMA connections, which stops the transfers, so the
daemons know that memory is safe to reuse once the helper is gone.

Run it as: python -m llmd_kv_tiering.rdma_helper --daemon-socket S --listen-socket L
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import os
import socket
import threading
import time

from llmd_kv_tiering.client import DaemonClient, DaemonError

logger = logging.getLogger("llmd_kv_tiering.rdma_helper")

HELPER_NAMESPACE = "__rdma_helper__"


def coalesce(pairs: list[tuple[int, int]], slot_bytes: int) -> list[tuple[int, int, int]]:
    """Merge runs where both the local and the remote slot advance by one.

    Returns (local_offset, remote_offset, length) in bytes. Fewer, larger
    descriptors make for fewer RDMA work requests.
    """
    out: list[tuple[int, int, int]] = []
    for local, remote in sorted(pairs):
        lo, ro = local * slot_bytes, remote * slot_bytes
        if out and out[-1][0] + out[-1][2] == lo and out[-1][1] + out[-1][2] == ro:
            prev = out[-1]
            out[-1] = (prev[0], prev[1], prev[2] + slot_bytes)
        else:
            out.append((lo, ro, slot_bytes))
    return out


def split_runs(runs: list[tuple[int, int, int]], parts: int) -> list[list[tuple[int, int, int]]]:
    """Split runs into at most `parts` groups of about equal byte count.

    A run longer than its share is cut, so one large contiguous read still
    spreads over several transfers.
    """
    total = sum(n for _lo, _ro, n in runs)
    if parts <= 1 or total == 0:
        return [runs] if runs else []
    share = -(-total // parts)  # ceiling
    groups: list[list[tuple[int, int, int]]] = [[]]
    room = share
    for lo, ro, n in runs:
        while n > 0:
            take = min(n, room)
            groups[-1].append((lo, ro, take))
            lo, ro, n, room = lo + take, ro + take, n - take, room - take
            if room == 0 and len(groups) < parts:
                groups.append([])
                room = share
    return [g for g in groups if g]


def coalesce_to_addrs(pairs: list[tuple[int, int]], slot_bytes: int) -> list[tuple[int, int, int]]:
    """Merge runs where the local slot advances by one and the remote
    address by one slot. Returns (local_offset, remote_address, length)."""
    out: list[tuple[int, int, int]] = []
    for local, addr in sorted(pairs):
        lo = local * slot_bytes
        if out and out[-1][0] + out[-1][2] == lo and out[-1][1] + out[-1][2] == addr:
            prev = out[-1]
            out[-1] = (prev[0], prev[1], prev[2] + slot_bytes)
        else:
            out.append((lo, addr, slot_bytes))
    return out


class Helper:
    def __init__(
        self,
        daemon_socket: str,
        name: str,
        backends: list[str],
        timeout_s: float,
        parallel: int = 8,
        num_threads: int = 0,
    ) -> None:
        self.name = name
        self.timeout = timeout_s
        # A read is split into this many concurrent NIXL transfers. One
        # transfer drives one queue pair and does not fill a fast NIC.
        self.parallel = max(1, parallel)
        self._lock = threading.Lock()
        self._peer_by_meta: dict[str, str] = {}
        self.client = self._attach(daemon_socket)
        arena = self.client.arena
        self.base, self.size, self.slot_bytes = arena.base, arena.size, arena.slot_bytes

        from nixl._api import nixl_agent, nixl_agent_config

        # Older NIXL releases (0.3, in the TPU image) have no num_threads
        # option; pass it only when it is set.
        cfg = {"num_threads": num_threads} if num_threads > 0 else {}
        self.agent = nixl_agent(name, nixl_agent_config(backends=backends, **cfg))
        self._reg = self.agent.register_memory([(self.base, self.size, 0, "")], "DRAM")
        self.metadata = self.agent.get_agent_metadata()
        logger.info(
            "RDMA helper %s: registered %d bytes of pool at 0x%x (slot %d bytes) with %s",
            name,
            self.size,
            self.base,
            self.slot_bytes,
            backends,
        )

    def _attach(self, daemon_socket: str) -> DaemonClient:
        # The pool exists once the first engine has connected; wait for it.
        while True:
            try:
                return DaemonClient(
                    daemon_socket,
                    HELPER_NAMESPACE,
                    0,
                    f"rdma-helper/{self.name}",
                    on_event=lambda _msg: None,
                    on_disconnect=self._lost,
                    timeout_s=self.timeout,
                )
            except (OSError, DaemonError) as e:
                logger.info("waiting for the daemon's pool: %s", e)
                time.sleep(2)

    def _lost(self) -> None:
        # The pool mapping stays valid, but a restarted daemon has a new pool.
        logger.error("lost the daemon; exiting so the helper restarts with it")
        os._exit(1)

    def info(self) -> dict:
        return {
            "name": self.name,
            "metadata": base64.b64encode(self.metadata).decode(),
            "base": self.base,
            "slot_bytes": self.slot_bytes,
        }

    def _peer(self, metadata_b64: str) -> str:
        meta = base64.b64decode(metadata_b64)
        digest = hashlib.sha256(meta).hexdigest()
        name = self._peer_by_meta.get(digest)
        if name is None:
            added = self.agent.add_remote_agent(meta)
            name = added.decode() if isinstance(added, bytes) else added
            self._peer_by_meta[digest] = name
        return name

    def _run(self, op: str, runs: list[tuple[int, int, int]], peer: str, deadline: float) -> dict:
        """Run `op` over (local_offset, remote_address, length) runs, in
        parallel groups, and wait until every started transfer is DONE or
        ERR. Returns {"unsafe": True} when that cannot be confirmed."""
        handles = []
        error = None
        for group in split_runs(runs, self.parallel):
            try:
                local = self.agent.get_xfer_descs([(self.base + lo, n, 0) for lo, _ra, n in group], "DRAM")
                remote = self.agent.get_xfer_descs([(ra, n, 0) for _lo, ra, n in group], "DRAM")
                handle = self.agent.initialize_xfer(op, local, remote, peer)
            except Exception as e:  # noqa: BLE001 - stop starting, wait for the rest
                error = f"RDMA {op} setup failed: {e}"
                break
            handles.append(handle)
            try:
                started = self.agent.transfer(handle)
            except Exception as e:  # noqa: BLE001 - its state is unknown
                return {"error": f"RDMA {op} start raised: {e}", "unsafe": True}
            if started == "ERR":
                error = f"RDMA {op} failed to start"
                break
        # Wait for every started transfer, even after a failure: freeing a
        # handle does not stop its DMA.
        pending = list(handles)
        while pending:
            if time.monotonic() > deadline:
                return {"error": f"RDMA {op} timed out with {len(pending)} transfers running", "unsafe": True}
            still = []
            for handle in pending:
                try:
                    state = self.agent.check_xfer_state(handle)
                except Exception as e:  # noqa: BLE001 - its state is unknown
                    return {"error": f"RDMA {op} state check raised: {e}", "unsafe": True}
                if state == "ERR":
                    error = error or f"RDMA {op} failed"
                elif state != "DONE":
                    still.append(handle)
            pending = still
        for handle in handles:
            self.agent.release_xfer_handle(handle)
        return {"error": error} if error else {}

    def _deadline(self, req: dict, received: float) -> float:
        timeout = req.get("timeout_s") or self.timeout
        return received + float(timeout)

    def _locked(self, deadline: float) -> bool:
        """Take the transfer lock, unless the deadline passes first."""
        return self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    def write(self, req: dict, received: float) -> dict:
        deadline = self._deadline(req, received)
        slot_bytes = int(req["slot_bytes"])
        if slot_bytes != self.slot_bytes:
            return {"error": f"slot size {slot_bytes} does not match local {self.slot_bytes}"}
        runs = coalesce_to_addrs([(int(local), int(addr)) for local, addr in req["pairs"]], slot_bytes)
        for lo, _ra, n in runs:
            if lo + n > self.size:
                return {"error": "local slot outside the pool"}
        if not self._locked(deadline):
            return {"error": "RDMA WRITE timed out before it started"}
        try:
            peer = self._peer(req["peer_metadata"])
            return self._run("WRITE", runs, peer, deadline)
        finally:
            self._lock.release()

    def read(self, req: dict, received: float) -> dict:
        deadline = self._deadline(req, received)
        slot_bytes = int(req["slot_bytes"])
        if slot_bytes != self.slot_bytes:
            return {"error": f"slot size {slot_bytes} does not match local {self.slot_bytes}"}
        remote_base = int(req["remote_base"])
        runs = coalesce([(int(local), int(remote)) for local, remote in req["pairs"]], slot_bytes)
        for lo, _ro, n in runs:
            if lo + n > self.size:
                return {"error": "local slot outside the pool"}
        if not self._locked(deadline):
            return {"error": "RDMA READ timed out before it started"}
        try:
            peer = self._peer(req["peer_metadata"])
            return self._run("READ", [(lo, remote_base + ro, n) for lo, ro, n in runs], peer, deadline)
        finally:
            self._lock.release()


def serve(helper: Helper, path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o660)
    srv.listen(16)
    logger.info("RDMA helper listening on %s", path)

    def handle(conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                received = time.monotonic()
                try:
                    req = json.loads(line)
                    op = req.get("op")
                    if op == "info":
                        resp = helper.info()
                    elif op == "read":
                        resp = helper.read(req, received)
                    elif op == "write":
                        resp = helper.write(req, received)
                    else:
                        resp = {"error": f"unknown op {op!r}"}
                except Exception as e:  # noqa: BLE001 - every failure becomes a response
                    logger.exception("request failed")
                    # Transfers are started only inside _run, which reports
                    # its own failures; an exception elsewhere started none.
                    resp = {"error": str(e)}
                with contextlib.suppress(OSError):
                    f.write(json.dumps(resp).encode() + b"\n")
                    f.flush()
                if resp.get("unsafe"):
                    # Transfers may still be running. Exiting closes this
                    # process's RDMA connections, which stops them; the
                    # helper is restarted with its container.
                    logger.critical("exiting: %s", resp.get("error"))
                    os._exit(1)

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--daemon-socket", required=True)
    ap.add_argument("--listen-socket", required=True)
    ap.add_argument("--name", default=f"kvtier-{socket.gethostname()}")
    ap.add_argument("--backends", default="UCX", help="comma-separated NIXL backends")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--parallel", type=int, default=8, help="concurrent NIXL transfers per read")
    ap.add_argument("--num-threads", type=int, default=0, help="NIXL backend threads (0 = NIXL default)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    helper = Helper(a.daemon_socket, a.name, a.backends.split(","), a.timeout_s, a.parallel, a.num_threads)
    serve(helper, a.listen_socket)


if __name__ == "__main__":
    main()
