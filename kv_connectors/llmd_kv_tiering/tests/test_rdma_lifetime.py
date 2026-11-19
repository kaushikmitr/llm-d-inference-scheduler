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
"""Transfer lifetime in the RDMA helper and the zero-copy reader, against a
fake NIXL agent. Real RDMA completion and failure behavior is not covered."""

from __future__ import annotations

import threading
import time

import pytest
from llmd_kv_tiering.client import DaemonError
from llmd_kv_tiering.rdma_helper import Helper
from llmd_kv_tiering.zero_copy import ZeroCopyReader


class FakeAgent:
    """Transfers finish after `polls` state checks. A transfer listed in
    fail_start fails to start; polls=None means it never finishes."""

    def __init__(self, polls=2, fail_start=()):
        self.polls = polls
        self.fail_start = set(fail_start)
        self.started = 0
        self.checks: dict[int, int] = {}
        self.released: list[int] = []
        self.released_while_running: list[int] = []

    def get_xfer_descs(self, descs, mem):
        return descs

    def initialize_xfer(self, op, local, remote, peer):
        self.started += 1
        return self.started

    def transfer(self, handle):
        return "ERR" if handle in self.fail_start else "PROC"

    def check_xfer_state(self, handle):
        if handle in self.fail_start:
            return "ERR"
        self.checks[handle] = self.checks.get(handle, 0) + 1
        if self.polls is not None and self.checks[handle] >= self.polls:
            return "DONE"
        return "PROC"

    def release_xfer_handle(self, handle):
        if handle not in self.fail_start and (self.polls is None or self.checks.get(handle, 0) < self.polls):
            self.released_while_running.append(handle)
        self.released.append(handle)


def helper(agent, parallel=4) -> Helper:
    h = Helper.__new__(Helper)
    h.agent, h.base, h.parallel, h.timeout = agent, 0, parallel, 5.0
    h.size, h.slot_bytes = 1 << 30, 0
    h._lock = threading.Lock()
    return h


RUNS = [(0, 1 << 20, 4 << 20)]  # one 4 MiB run, split into 4 transfers


def test_waits_for_started_transfers_after_a_start_failure():
    agent = FakeAgent(polls=3, fail_start={3})
    resp = helper(agent)._run("WRITE", RUNS, "peer", time.monotonic() + 5)
    # Transfers 1 and 2 started before 3 failed. Both ran to DONE before
    # their handles were freed, and the error is clean: nothing is running.
    assert resp == {"error": "RDMA WRITE failed to start"}
    assert agent.started == 3
    assert agent.released_while_running == []
    assert sorted(agent.released) == [1, 2, 3]


def test_deadline_with_transfers_running_is_unsafe():
    agent = FakeAgent(polls=None)
    resp = helper(agent)._run("READ", RUNS, "peer", time.monotonic() + 0.05)
    assert resp["unsafe"] is True
    # Handles of running transfers are not freed; the helper exits instead.
    assert agent.released == []


def test_a_failed_transfer_still_waits_for_the_others():
    agent = FakeAgent(polls=3)
    agent.check_xfer_state_orig = agent.check_xfer_state

    def check(handle):
        return "ERR" if handle == 1 else agent.check_xfer_state_orig(handle)

    agent.check_xfer_state = check
    resp = helper(agent)._run("WRITE", RUNS, "peer", time.monotonic() + 5)
    assert resp == {"error": "RDMA WRITE failed"}
    assert all(agent.checks.get(h, 0) >= 3 for h in (2, 3, 4))


def test_busy_helper_does_not_start_after_the_deadline():
    h = helper(FakeAgent())
    h._lock.acquire()
    try:
        resp = h.write({"slot_bytes": 0, "pairs": [], "timeout_s": 0.05}, time.monotonic())
    finally:
        h._lock.release()
    assert resp == {"error": "RDMA WRITE timed out before it started"}


class FakeClient:
    def __init__(self, error):
        self.error = error

    def call(self, op, **kw):
        raise DaemonError(self.error)


@pytest.mark.parametrize(("error", "reregistered"), [("no_rdma", False), ("rdma transfer state unknown", True)])
def test_zero_copy_reregisters_after_an_unknown_failure(monkeypatch, error, reregistered):
    deregistered = []

    class Agent:
        def __init__(self, gen):
            self.gen = gen

        def deregister_memory(self, reg):
            deregistered.append(self.gen)

    reader = ZeroCopyReader.__new__(ZeroCopyReader)
    reader._base, reader._size, reader._block, reader._name, reader._generation = 0, 1 << 20, 4096, "t", 0

    def register():
        return Agent(reader._generation), "reg", f"meta-{reader._generation}".encode()

    monkeypatch.setattr(reader, "_register", register)
    reader._agent, reader._reg, reader._metadata = register()
    with pytest.raises(DaemonError):
        reader.read(FakeClient(error), "tcp://src:1", [b"k"], [0])
    assert (reader._generation == 1) is reregistered
    assert deregistered == ([0] if reregistered else [])
