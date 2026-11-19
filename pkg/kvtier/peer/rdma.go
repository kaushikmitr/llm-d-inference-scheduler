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

package peer

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
)

// RDMAHelper is a client for the node's RDMA helper process
// (llmd_kv_tiering.rdma_helper). The helper maps the daemon's pool,
// registers it with NIXL, and performs RDMA reads from peer pools into
// local slots. Requests and responses are JSON lines over a Unix socket.
type RDMAHelper struct {
	path string

	mu   sync.Mutex
	conn net.Conn
	rd   *bufio.Reader
	info *RDMAInfo
}

// RDMAInfo identifies a helper to its peers.
type RDMAInfo struct {
	Name      string `json:"name"`
	Metadata  []byte `json:"metadata"`
	Base      uint64 `json:"base"`
	SlotBytes uint64 `json:"slot_bytes"`
}

type rdmaRequest struct {
	Op string `json:"op"`
	// TimeoutS is the helper's deadline for the request, counted from when
	// it receives it. Every transfer it starts ends by then, or it exits.
	TimeoutS   float64    `json:"timeout_s,omitempty"`
	Peer       string     `json:"peer,omitempty"`
	PeerMeta   []byte     `json:"peer_metadata,omitempty"`
	RemoteBase uint64     `json:"remote_base,omitempty"`
	SlotBytes  uint64     `json:"slot_bytes,omitempty"`
	Pairs      [][2]int64 `json:"pairs,omitempty"` // [local slot, remote slot]
}

type rdmaResponse struct {
	Error string `json:"error,omitempty"`
	// Unsafe means the helper could not confirm that every transfer it
	// started has stopped. It exits after sending this, which stops them.
	Unsafe bool `json:"unsafe,omitempty"`
	RDMAInfo
}

// rdmaMargin is the slack between the deadlines of the parties to an RDMA
// transfer. The helper ends every transfer, or exits, within the transfer
// timeout T. Its daemon waits T+margin for it. The other daemon waits
// T+2*margin. So a party that gives up knows the transfer has stopped.
const rdmaMargin = 10 * time.Second

// ErrRDMAUnsafe means an RDMA transfer failed in a way that leaves its
// memory state unknown: a transfer may still be running. Memory it could
// write must not be reused until the helper that ran it has exited, which
// happens within the transfer timeout.
var ErrRDMAUnsafe = errors.New("rdma transfer state unknown")

// helperWait is how long a daemon waits for its own helper.
func (n *Node) helperWait() time.Duration { return n.opts.Timeout + rdmaMargin }

// peerWait is how long a daemon waits for the other daemon in an RDMA
// transfer.
func (n *Node) peerWait() time.Duration { return n.opts.Timeout + 2*rdmaMargin }

// NewRDMAHelper returns a client for the helper at path. It connects on
// first use.
func NewRDMAHelper(path string) *RDMAHelper {
	return &RDMAHelper{path: path}
}

func (h *RDMAHelper) call(ctx context.Context, req *rdmaRequest, timeout time.Duration) (*rdmaResponse, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.conn == nil {
		d := net.Dialer{Timeout: timeout}
		conn, err := d.DialContext(ctx, "unix", h.path)
		if err != nil {
			return nil, fmt.Errorf("rdma helper: %w", err)
		}
		h.conn = conn
		h.rd = bufio.NewReader(conn)
	}
	fail := func(err error) error {
		if cerr := h.conn.Close(); cerr != nil {
			err = errors.Join(err, cerr)
		}
		h.conn, h.rd = nil, nil
		return fmt.Errorf("rdma helper: %w", err)
	}
	if err := h.conn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return nil, fail(err)
	}
	line, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	// Once a request may have reached the helper, a failure without a
	// clean answer leaves the transfer's state unknown.
	unsafe := func(err error) error {
		if req.Op == "info" {
			return err
		}
		return fmt.Errorf("%w: %w", ErrRDMAUnsafe, err)
	}
	if _, err := h.conn.Write(append(line, '\n')); err != nil {
		return nil, unsafe(fail(err))
	}
	out, err := h.rd.ReadBytes('\n')
	if err != nil {
		return nil, unsafe(fail(err))
	}
	resp := &rdmaResponse{}
	if err := json.Unmarshal(out, resp); err != nil {
		return nil, unsafe(fail(err))
	}
	if resp.Unsafe {
		h.info = nil // a restarted helper has new metadata
		return nil, unsafe(fail(errors.New(resp.Error)))
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("rdma helper: %s", resp.Error)
	}
	return resp, nil
}

// Info returns the helper's identity, cached after the first call.
func (h *RDMAHelper) Info(ctx context.Context, timeout time.Duration) (*RDMAInfo, error) {
	h.mu.Lock()
	cached := h.info
	h.mu.Unlock()
	if cached != nil {
		return cached, nil
	}
	resp, err := h.call(ctx, &rdmaRequest{Op: "info"}, timeout)
	if err != nil {
		return nil, err
	}
	info := resp.RDMAInfo
	h.mu.Lock()
	h.info = &info
	h.mu.Unlock()
	return &info, nil
}

// Read copies remote slots into local slots with RDMA reads.
func (h *RDMAHelper) Read(ctx context.Context, peer string, peerMeta []byte, remoteBase, slotBytes uint64,
	pairs [][2]int64, timeout time.Duration,
) error {
	_, err := h.call(ctx, &rdmaRequest{
		Op: "read", Peer: peer, PeerMeta: peerMeta, RemoteBase: remoteBase,
		SlotBytes: slotBytes, Pairs: pairs, TimeoutS: timeout.Seconds(),
	}, timeout+rdmaMargin)
	return err
}

// serveRDMA answers an RDMA peer_get: it sends the slots of the found
// blocks and this helper's identity, then waits for the destination's
// peer_release before it drops the pins. It returns handled=false when the
// helper is unavailable, so the caller can send the bytes over TCP.
func (n *Node) serveRDMA(conn net.Conn, req, resp *protocol.Message, slots []int64, found [][]byte,
	arena *pool.Arena, sid pool.SessionID,
) (bool, error) {
	info, err := n.rdma.Info(context.Background(), n.opts.Timeout)
	if err != nil {
		n.log.Warn("rdma unavailable, serving over TCP", "err", err)
		return false, nil
	}
	if info.SlotBytes != arena.SlotBytes() {
		n.log.Warn("rdma helper slot size differs from the pool, serving over TCP",
			"helper", info.SlotBytes, "pool", arena.SlotBytes())
		return false, nil
	}
	resp.RDMA = true
	resp.Slots = slots
	resp.Agent = info.Name
	resp.AgentMeta = info.Metadata
	resp.Base = info.Base
	if err := writeFrame(conn, resp); err != nil {
		return true, err
	}
	// The destination reads the slots now. Keep them pinned until it says
	// it is done. It waits at most helperWait for its helper, so by
	// peerWait its read has finished, or the destination has discarded it.
	if err := conn.SetReadDeadline(time.Now().Add(n.peerWait())); err != nil {
		return true, err
	}
	rel, err := protocol.Read(conn)
	if err != nil {
		return true, fmt.Errorf("waiting for peer_release: %w", err)
	}
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
		return true, err
	}
	if rel.Op != protocol.OpPeerRelease {
		return true, fmt.Errorf("expected %s, got %s", protocol.OpPeerRelease, rel.Op)
	}
	n.store.Unpin(sid, req.Namespace, found)
	if rel.Error != "" {
		return true, nil // the destination's read failed; keep the blocks
	}
	n.Served.Add(uint64(len(found)))
	n.BytesOut.Add(uint64(len(found)) * arena.SlotBytes())
	if req.Evict && len(found) > 0 {
		n.remove(req.Namespace, found)
	}
	return true, nil
}

// rdmaReady reports whether this node's RDMA helper answers. Without it,
// fetches use TCP.
func (n *Node) rdmaReady(ctx context.Context) bool {
	if n.rdma == nil {
		return false
	}
	if _, err := n.rdma.Info(ctx, n.opts.Timeout); err != nil {
		n.log.Debug("rdma helper not ready, fetching over TCP", "err", err)
		return false
	}
	return true
}

// fetchRDMA asks the source for want in RDMA mode and reads the blocks
// from its pool into the reserved local slots. A source without an RDMA
// helper answers with the bytes over TCP instead, which is read here too.
func (n *Node) fetchRDMA(ctx context.Context, endpoint, ns string, want [][]byte, wantIdx []int,
	slots []int64, status []uint8, ok []bool, evict bool, arena *pool.Arena,
) error {
	conn, err := n.get(ctx, endpoint)
	if err != nil {
		return err
	}
	deadline := time.Now().Add(n.peerWait())
	if err := conn.SetDeadline(deadline); err != nil {
		n.put(endpoint, conn, false)
		return err
	}
	healthy := false
	defer func() { n.put(endpoint, conn, healthy) }()

	req := &protocol.Message{
		ID: 1, Op: protocol.OpPeerGet, Namespace: ns, Keys: want, Evict: evict,
		SlotBytes: arena.SlotBytes(), RDMA: true,
	}
	if err := writeFrame(conn, req); err != nil {
		return err
	}
	resp, err := protocol.Read(conn)
	if err != nil {
		return err
	}
	if resp.Error != "" {
		healthy = true
		return fmt.Errorf("source %s: %s", endpoint, resp.Error)
	}
	if len(resp.Status) != len(want) {
		return fmt.Errorf("source %s returned %d statuses for %d keys", endpoint, len(resp.Status), len(want))
	}
	if !resp.RDMA {
		// The source has no RDMA helper: the bytes follow on this stream.
		for j, st := range resp.Status {
			i := wantIdx[j]
			if st != protocol.FetchOK {
				status[i] = protocol.FetchMissing
				continue
			}
			if _, err := io.ReadFull(conn, arena.Slot(slots[i])); err != nil {
				status[i] = protocol.FetchFailed
				return err
			}
			n.BytesIn.Add(arena.SlotBytes())
			ok[j] = true
			status[i] = protocol.FetchOK
		}
		healthy = true
		return nil
	}

	var pairs [][2]int64
	var idx []int
	for j, st := range resp.Status {
		i := wantIdx[j]
		if st != protocol.FetchOK || j >= len(resp.Slots) || resp.Slots[j] < 0 {
			status[i] = protocol.FetchMissing
			continue
		}
		pairs = append(pairs, [2]int64{slots[i], resp.Slots[j]})
		idx = append(idx, j)
	}
	readErr := error(nil)
	if len(pairs) > 0 {
		readErr = n.rdma.Read(ctx, resp.Agent, resp.AgentMeta, resp.Base, arena.SlotBytes(), pairs, n.opts.Timeout)
	}
	release := &protocol.Message{ID: 2, Op: protocol.OpPeerRelease}
	if readErr != nil {
		release.Error = readErr.Error()
	}
	if err := writeFrame(conn, release); err != nil {
		return errors.Join(readErr, err)
	}
	if readErr != nil {
		for _, j := range idx {
			status[wantIdx[j]] = protocol.FetchFailed
		}
		healthy = true
		return readErr
	}
	for _, j := range idx {
		ok[j] = true
		status[wantIdx[j]] = protocol.FetchOK
		n.BytesIn.Add(arena.SlotBytes())
	}
	healthy = true
	return nil
}

// ErrNoRemoteRDMA means the source cannot push blocks with RDMA. The caller
// should fetch them into its pool instead.
var ErrNoRemoteRDMA = errors.New("source does not serve RDMA")

// errUnsafeCode is the error a source sends when a push failed with its
// state unknown. The destination maps it to ErrRDMAUnsafe.
const errUnsafeCode = "rdma_unsafe"

// Write copies local slots to remote addresses with RDMA writes. Each pair
// is [local slot, remote address].
func (h *RDMAHelper) Write(ctx context.Context, peerMeta []byte, slotBytes uint64, pairs [][2]int64,
	timeout time.Duration,
) error {
	_, err := h.call(ctx, &rdmaRequest{
		Op: "write", PeerMeta: peerMeta, SlotBytes: slotBytes, Pairs: pairs, TimeoutS: timeout.Seconds(),
	}, timeout+rdmaMargin)
	return err
}

// PushRemote asks the source daemon at endpoint to write keys straight into
// the caller's registered memory: key i goes to dstAddrs[i] in the agent
// described by dstMeta. When evict is set, the source drops the blocks it
// wrote. It returns a protocol.Fetch* status per key, once the writes are
// complete.
func (n *Node) PushRemote(ctx context.Context, endpoint, ns string, keys [][]byte, slotBytes uint64,
	dstMeta []byte, dstAddrs []uint64, evict bool,
) ([]uint8, error) {
	conn, err := n.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	if err := conn.SetDeadline(time.Now().Add(n.peerWait())); err != nil {
		n.put(endpoint, conn, false)
		return nil, err
	}
	req := &protocol.Message{
		ID: 1, Op: protocol.OpPeerGet, Namespace: ns, Keys: keys, SlotBytes: slotBytes,
		Push: true, DstMeta: dstMeta, DstAddrs: dstAddrs, Evict: evict,
	}
	if err := writeFrame(conn, req); err != nil {
		n.put(endpoint, conn, false)
		// Part of the request may have reached the source.
		return nil, fmt.Errorf("%w: %w", ErrRDMAUnsafe, err)
	}
	resp, err := protocol.Read(conn)
	if err != nil {
		n.put(endpoint, conn, false)
		return nil, fmt.Errorf("%w: no answer from %s: %w", ErrRDMAUnsafe, endpoint, err)
	}
	n.put(endpoint, conn, true)
	switch {
	case resp.Error == "no_rdma":
		return nil, ErrNoRemoteRDMA
	case resp.Error == errUnsafeCode:
		return nil, fmt.Errorf("%w: source %s", ErrRDMAUnsafe, endpoint)
	case resp.Error != "":
		return nil, fmt.Errorf("source %s: %s", endpoint, resp.Error)
	case len(resp.Status) != len(keys):
		return nil, fmt.Errorf("source %s returned %d statuses for %d keys", endpoint, len(resp.Status), len(keys))
	}
	return resp.Status, nil
}

// servePush answers a push peer_get: it writes the pinned blocks into the
// destination's memory with RDMA, then reports a status per key. It answers
// "no_rdma" when its helper is unavailable, so the destination can fetch
// some other way.
func (n *Node) servePush(conn net.Conn, req, resp *protocol.Message, slots []int64, found [][]byte,
	arena *pool.Arena, sid pool.SessionID,
) error {
	unpinned := false
	unpin := func() {
		if !unpinned {
			n.store.Unpin(sid, req.Namespace, found)
			unpinned = true
		}
	}
	defer unpin()
	if len(req.DstAddrs) != len(req.Keys) || !n.rdmaReady(context.Background()) {
		resp.Status = nil
		resp.Error = "no_rdma"
		return writeFrame(conn, resp)
	}
	var pairs [][2]int64
	for i, s := range slots {
		if s >= 0 {
			pairs = append(pairs, [2]int64{s, int64(req.DstAddrs[i])}) // #nosec G115 -- user-space addresses fit in int64
		}
	}
	if len(pairs) > 0 {
		if err := n.rdma.Write(context.Background(), req.DstMeta, arena.SlotBytes(), pairs, n.opts.Timeout); err != nil {
			pushFailed(resp, slots, err)
			n.log.Warn("rdma push failed", "err", err)
			return writeFrame(conn, resp)
		}
	}
	n.Served.Add(uint64(len(found)))
	n.BytesOut.Add(uint64(len(found)) * arena.SlotBytes())
	if req.Evict && len(found) > 0 {
		unpin()
		if removed := n.store.Remove(req.Namespace, found); len(removed) > 0 {
			n.notify.AnnounceRemoved(removed)
		}
	}
	return writeFrame(conn, resp)
}

// fetchPush has the source write want straight into the reserved local pool
// slots with RDMA WRITE. On 4 KiB-page memory a WRITE runs at about twice
// the bandwidth of a READ. It returns ErrNoRemoteRDMA when the source
// cannot push, before anything is transferred.
func (n *Node) fetchPush(ctx context.Context, endpoint, ns string, want [][]byte, wantIdx []int,
	slots []int64, status []uint8, ok []bool, evict bool, arena *pool.Arena,
) error {
	info, err := n.rdma.Info(ctx, n.opts.Timeout)
	if err != nil {
		return err
	}
	if info.SlotBytes != arena.SlotBytes() {
		return fmt.Errorf("rdma helper slot size %d differs from the pool's %d", info.SlotBytes, arena.SlotBytes())
	}
	addrs := make([]uint64, len(want))
	for j, i := range wantIdx {
		addrs[j] = info.Base + uint64(slots[i])*arena.SlotBytes() // #nosec G115 -- reserved slots are non-negative
	}
	st, err := n.PushRemote(ctx, endpoint, ns, want, arena.SlotBytes(), info.Metadata, addrs, evict)
	if err != nil {
		return err
	}
	for j, s := range st {
		i := wantIdx[j]
		status[i] = s
		if s == protocol.FetchOK {
			ok[j] = true
			n.BytesIn.Add(arena.SlotBytes())
		}
	}
	return nil
}

// pushFailed marks every found block of a failed push as failed. When the
// failure leaves writes into the destination's memory possibly in flight,
// it tells the destination, which must not reuse that memory until our
// helper, which exits on such a failure, is gone.
func pushFailed(resp *protocol.Message, slots []int64, err error) {
	for i, s := range slots {
		if s >= 0 {
			resp.Status[i] = protocol.FetchFailed
		}
	}
	if errors.Is(err, ErrRDMAUnsafe) {
		resp.Error = errUnsafeCode
	}
}
