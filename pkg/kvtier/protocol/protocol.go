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

// Package protocol is the wire format between the daemon and engine-side
// shims. Each frame is a 4-byte big-endian length followed by one msgpack
// map. The daemon passes the arena's file descriptor with SCM_RIGHTS on the
// hello response.
package protocol

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/vmihailenco/msgpack/v5"
)

// Version is the protocol version a client must send in hello.
const Version = 1

// MaxFrame bounds a frame's size.
const MaxFrame = 256 << 20

// Operations. A request carries a nonzero ID that the response echoes.
// Events pushed by the daemon carry ID 0.
const (
	OpHello    = "hello"
	OpAlloc    = "alloc"
	OpCommit   = "commit"
	OpPin      = "pin"
	OpUnpin    = "unpin"
	OpSnapshot = "snapshot"
	OpStats    = "stats"
	OpEvent    = "event"

	// Local requests that move blocks between daemons.
	OpFetch = "fetch" // pull keys from a peer into this daemon's pool
	OpPush  = "push"  // ask a peer to pull blocks, named by router hash, from us

	// Requests between daemons, over TCP.
	OpPeerGet   = "peer_get"   // send these blocks; raw block bytes follow the response
	OpPeerOffer = "peer_offer" // please pull these keys from the named endpoint
	// OpPeerHints is the direct call a router makes to a daemon. It carries
	// a KV hint envelope, the same one a request would carry, and the
	// daemon's control token.
	OpPeerHints = "peer_hints"
	// OpPeerRelease ends an RDMA peer_get: the destination has read the
	// slots, and the source may drop its pins.
	OpPeerRelease = "peer_release"

	// OpRemotePush asks a source daemon to RDMA-write blocks straight into
	// engine memory that the requesting engine registered with NIXL.
	OpRemotePush = "remote_push"
)

// Per-key outcome of a fetch.
const (
	FetchOK      uint8 = 0 // pulled into the pool
	FetchPresent uint8 = 1 // already in the pool
	FetchMissing uint8 = 2 // the source does not have it
	FetchNoSpace uint8 = 3 // no free or evictable slot here
	FetchFailed  uint8 = 4 // transfer error
)

// Versions and action types of the KV hint envelope. They match vLLM's
// KvHintsEnvelope, so a router sends a daemon the hint it would attach to a
// request.
const (
	HintProtocolVersion = "0.1"
	HintActionVersion   = "1.0"
	HintPrefetch        = "kv.prefetch"
	HintMove            = "kv.move"
	// Payload fields of kv.prefetch and kv.move.
	HintDestinationField = "destination_control_endpoint"
	HintBlockHashesField = "block_hashes"
)

// HintsEnvelope is vLLM's KV hint envelope.
type HintsEnvelope struct {
	ProtocolVersion string       `msgpack:"protocol_version"`
	MessageID       string       `msgpack:"message_id"`
	Actions         []HintAction `msgpack:"actions"`
}

// HintAction is one action of a hint envelope.
type HintAction struct {
	ActionID      string         `msgpack:"action_id"`
	ActionType    string         `msgpack:"action_type"`
	ActionVersion string         `msgpack:"action_version"`
	Payload       map[string]any `msgpack:"payload"`
}

// Message is the single envelope for every request, response and event.
// Each operation uses a subset of the fields.
type Message struct {
	ID    uint64 `msgpack:"id"`
	Op    string `msgpack:"op"`
	Error string `msgpack:"error,omitempty"`

	// hello request
	Version   int    `msgpack:"version,omitempty"`
	Namespace string `msgpack:"ns,omitempty"`
	SlotBytes uint64 `msgpack:"slot_bytes,omitempty"`
	Client    string `msgpack:"client,omitempty"`
	// hello response (the arena fd travels as SCM_RIGHTS)
	ArenaBytes uint64 `msgpack:"arena_bytes,omitempty"`
	NumSlots   int64  `msgpack:"num_slots,omitempty"`
	// PoolID identifies the pool. A restarted daemon has a new pool with a
	// new ID, so a client can tell slot numbers of different pools apart.
	PoolID  uint64 `msgpack:"pool_id,omitempty"`
	Session uint64 `msgpack:"session,omitempty"`
	// SelfEndpoint is this daemon's advertised peer endpoint, if any.
	SelfEndpoint string `msgpack:"self_endpoint,omitempty"`

	// alloc, commit, pin, unpin, snapshot
	Keys   [][]byte `msgpack:"keys,omitempty"`
	Slots  []int64  `msgpack:"slots,omitempty"`
	Status []uint8  `msgpack:"status,omitempty"`
	OK     []bool   `msgpack:"ok,omitempty"`

	// event, and the evictions caused by alloc
	Stored  [][]byte `msgpack:"stored,omitempty"`
	Removed [][]byte `msgpack:"removed,omitempty"`

	// fetch, push, peer_get, peer_offer
	Endpoint string            `msgpack:"endpoint,omitempty"` // the peer to pull from or offer to
	Hashes   []uint64          `msgpack:"hashes,omitempty"`   // router hashes (push)
	Evict    bool              `msgpack:"evict,omitempty"`    // the source drops what it sends
	Failures map[uint64]string `msgpack:"failures,omitempty"` // push: hash -> reason
	// peer_hints request: the control token, the model the hints are for,
	// and the hint envelope. Namespace, when set, replaces Model: it names
	// one namespace, or a prefix when it ends in "*".
	Token string         `msgpack:"token,omitempty"`
	Model string         `msgpack:"model,omitempty"`
	Hints *HintsEnvelope `msgpack:"kv_hints,omitempty"`
	// peer_hints response: the actions that were not run, by action ID,
	// with the reason. Failures lists the hashes the source did not hold.
	Rejected map[string]string `msgpack:"rejected,omitempty"`

	// RDMA peer_get: the source answers with slots and its RDMA helper's
	// identity instead of block bytes, and the destination reads the slots.
	RDMA      bool   `msgpack:"rdma,omitempty"`
	Agent     string `msgpack:"agent,omitempty"`      // source helper's NIXL agent name
	AgentMeta []byte `msgpack:"agent_meta,omitempty"` // source helper's NIXL metadata
	Base      uint64 `msgpack:"base,omitempty"`       // address of the source pool's slot 0

	// remote_push, and peer_get with Push: write the blocks into the
	// destination's registered memory at DstAddrs, one address per key.
	Push     bool     `msgpack:"push,omitempty"`
	DstMeta  []byte   `msgpack:"dst_meta,omitempty"`  // destination's NIXL metadata
	DstAddrs []uint64 `msgpack:"dst_addrs,omitempty"` // destination address per key

	// stats
	Stats map[string]any `msgpack:"stats,omitempty"`
}

// Encode returns m as a length-prefixed frame.
func Encode(m *Message) ([]byte, error) {
	body, err := msgpack.Marshal(m)
	if err != nil {
		return nil, err
	}
	if len(body) > MaxFrame {
		return nil, fmt.Errorf("frame of %d bytes exceeds limit", len(body))
	}
	buf := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(buf, uint32(len(body))) // #nosec G115 -- len(body) <= MaxFrame, checked above
	copy(buf[4:], body)
	return buf, nil
}

// Read reads one frame from r.
func Read(r io.Reader) (*Message, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n > MaxFrame {
		return nil, fmt.Errorf("frame of %d bytes exceeds limit", n)
	}
	body := make([]byte, n)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	m := &Message{}
	if err := msgpack.Unmarshal(body, m); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return m, nil
}
