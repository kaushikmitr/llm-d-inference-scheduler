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
"""Router fetch hints for vLLM on TPU with vllm-torchtpu.

vllm-torchtpu already offloads KV blocks to TPU Sync's host store, reads
them from peers through the global registry, and reports evictions as KV
events. This module adds one thing: a peer's blocks are read only when the
request carries a ``kv.fetch`` hint that names that peer.

Load it with no vllm-torchtpu change:

    --kv-transfer-config '{
      "kv_connector": "TPURaidenOffloadingConnector",
      "kv_connector_module_path": "llmd_kv_tiering.torchtpu_connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "global_registry_address": "registry-host:50051",
        "store_server_ip": "<pod IP>",
        "raiden_job_name": "<pod IP>",
        "llmd_peer_read": "hint"
      }
    }'

``raiden_job_name`` must be the pod IP. A store is named by its job name,
and a hint names a pod, so this is what ties the two together.

``llmd_peer_read`` is ``hint`` (the default) or ``always``. ``always`` keeps
vllm-torchtpu's own behavior: every local miss asks the registry.

Not handled here: ``kv.prefetch`` and ``kv.move``. A push shares the store's
save results and pin accounting with vllm-torchtpu's manager, so it belongs
inside that manager.
"""

from __future__ import annotations

import logging

from vllm_torchtpu.offload.raiden_connector import (
    TPURaidenOffloadingConnector as _BaseConnector,
)

from llmd_kv_tiering.peer_gate import FetchGate

logger = logging.getLogger("vllm.llmd_kv_tiering.torchtpu")


class LLMDTPURaidenOffloadingConnector(_BaseConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._llmd_gate: FetchGate | None = None
        if self.connector_scheduler is not None:
            extra = dict(getattr(vllm_config.kv_transfer_config, "kv_connector_extra_config", None) or {})
            mode = extra.get("llmd_peer_read", "hint")
            self._llmd_gate = FetchGate(self.connector_scheduler._raiden_manager, mode)
            logger.info("llm-d fetch gate on: peer reads %s", "need a kv.fetch hint" if mode == "hint" else "always")


# vLLM gives kv_connector_module_path priority over its registry, and
# vllm-torchtpu enables its offloading patches by connector name. Exporting
# the subclass under the base class's name keeps both working.
TPURaidenOffloadingConnector = LLMDTPURaidenOffloadingConnector
