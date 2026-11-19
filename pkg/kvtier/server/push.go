// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"context"
	"errors"

	"github.com/llm-d/llm-d-router/pkg/kvtier/peer"
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
)

// remotePush asks the source daemon to RDMA-write req.Keys straight into the
// requesting engine's registered memory, at req.DstAddrs. The blocks never
// enter this daemon's pool.
func (s *Server) remotePush(ss *session, req, resp *protocol.Message) {
	if s.peer == nil {
		resp.Error = "peer transfer is disabled on this daemon"
		return
	}
	arena := s.store.Arena()
	if arena == nil {
		resp.Error = "pool not initialised"
		return
	}
	if len(req.DstAddrs) != len(req.Keys) {
		resp.Error = "need one destination address per key"
		return
	}
	status, err := s.peer.PushRemote(context.Background(), req.Endpoint, ss.ns, req.Keys,
		arena.SlotBytes(), req.DstMeta, req.DstAddrs, req.Evict)
	switch {
	case errors.Is(err, peer.ErrNoRemoteRDMA):
		resp.Error = "no_rdma"
	case err != nil:
		resp.Error = err.Error()
	default:
		resp.Status = status
	}
}
