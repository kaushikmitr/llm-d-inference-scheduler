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

// Package peer moves KV blocks between daemons over TCP.
//
// Transfers are pulled by the destination: the destination
// reserves slots, asks the source for the blocks, and reads the bytes
// straight into those slots. A peer_get response is a msgpack frame with a
// per-key status, followed by the raw bytes of every found block in key
// order.
package peer

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
)

// Notifier announces pool changes to local engine sessions.
type Notifier interface {
	// Ordered runs fn under the lock that orders pool changes and their
	// events. A pool change and the Announce calls for it run inside one
	// Ordered call, so engines see changes in the order they happened.
	Ordered(fn func())
	AnnounceStored(ns string, refs []pool.KeyRef)
	AnnounceRemoved(refs []pool.KeyRef)
}

// Options configure a Node.
type Options struct {
	// ListenAddr is the TCP address to accept peers on, e.g. ":9401".
	ListenAddr string
	// Endpoint is the address other daemons use to reach this one, e.g.
	// "tcp://10.0.0.5:9401". Routers put it in hints.
	Endpoint string
	// Timeout bounds one transfer.
	Timeout time.Duration
	// MaxIdleConns is the number of idle connections kept per peer.
	MaxIdleConns int
	// Streams is the number of TCP connections one fetch spreads its
	// blocks over.
	Streams int
	// RDMAHelperSocket is the Unix socket of this node's RDMA helper. When
	// set, fetches read blocks from the source pool with RDMA, and this
	// daemon serves its pool to peers that ask for RDMA. Empty means TCP
	// only.
	RDMAHelperSocket string
	// ControlToken authorises peer_push, the direct call a router makes to
	// move blocks. Empty disables peer_push.
	ControlToken string
}

// Node is one daemon's peer endpoint: a server for other daemons and a
// client to them.
type Node struct {
	store  *pool.Store
	notify Notifier
	log    *slog.Logger
	opts   Options

	ln      net.Listener
	wg      sync.WaitGroup
	closing atomic.Bool

	sessions atomic.Uint64 // internal pool session ids

	mu    sync.Mutex
	idle  map[string][]net.Conn
	conns map[net.Conn]struct{}

	BytesIn  atomic.Uint64
	BytesOut atomic.Uint64
	Fetched  atomic.Uint64
	Served   atomic.Uint64

	rdma *RDMAHelper // nil when RDMA is off
}

// New returns a node. Call Listen to accept peers.
func New(store *pool.Store, notify Notifier, log *slog.Logger, opts Options) *Node {
	if opts.Timeout == 0 {
		opts.Timeout = 30 * time.Second
	}
	if opts.Streams <= 0 {
		opts.Streams = 8
	}
	if opts.MaxIdleConns < opts.Streams {
		opts.MaxIdleConns = opts.Streams
	}
	node := &Node{
		store:  store,
		notify: notify,
		log:    log,
		opts:   opts,
		idle:   make(map[string][]net.Conn),
		conns:  make(map[net.Conn]struct{}),
	}
	if opts.RDMAHelperSocket != "" {
		node.rdma = NewRDMAHelper(opts.RDMAHelperSocket)
	}
	return node
}

// Endpoint returns the advertised endpoint.
func (n *Node) Endpoint() string { return n.opts.Endpoint }

// session returns a fresh pool session id. The high bit keeps it apart from
// the ids of engine sessions.
func (n *Node) session() pool.SessionID {
	return pool.SessionID(1<<63 | n.sessions.Add(1))
}

// HostPort strips an optional tcp:// scheme from an endpoint.
func HostPort(endpoint string) string {
	return strings.TrimPrefix(endpoint, "tcp://")
}

func (n *Node) isSelf(endpoint string) bool {
	return n.opts.Endpoint != "" && HostPort(endpoint) == HostPort(n.opts.Endpoint)
}

// Listen starts accepting peers.
func (n *Node) Listen() error {
	var lc net.ListenConfig
	ln, err := lc.Listen(context.Background(), "tcp", n.opts.ListenAddr)
	if err != nil {
		return err
	}
	n.ln = ln
	if n.opts.Endpoint == "" {
		n.opts.Endpoint = "tcp://" + ln.Addr().String()
	}
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			n.track(c, true)
			n.wg.Add(1)
			go func() {
				defer n.wg.Done()
				defer n.track(c, false)
				n.serve(c)
			}()
		}
	}()
	return nil
}

// Addr returns the bound address.
func (n *Node) Addr() net.Addr { return n.ln.Addr() }

func (n *Node) track(c net.Conn, add bool) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if add {
		n.conns[c] = struct{}{}
	} else {
		delete(n.conns, c)
		c.Close()
	}
}

// Close stops the node and closes every connection.
func (n *Node) Close() {
	n.closing.Store(true)
	if n.ln != nil {
		n.ln.Close()
	}
	n.mu.Lock()
	for c := range n.conns {
		c.Close()
	}
	for _, cs := range n.idle {
		for _, c := range cs {
			c.Close()
		}
	}
	n.idle = map[string][]net.Conn{}
	n.mu.Unlock()
	n.wg.Wait()
}

// ---------------------------------------------------------------------------
// Serving other daemons
// ---------------------------------------------------------------------------

func (n *Node) serve(conn net.Conn) {
	for {
		req, err := protocol.Read(conn)
		if err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, net.ErrClosed) && !n.closing.Load() {
				n.log.Debug("peer read", "remote", conn.RemoteAddr(), "err", err)
			}
			return
		}
		if err := conn.SetWriteDeadline(time.Now().Add(n.opts.Timeout)); err != nil {
			n.log.Debug("peer set deadline", "remote", conn.RemoteAddr(), "err", err)
			return
		}
		switch req.Op {
		case protocol.OpPeerGet:
			err = n.serveGet(conn, req)
		case protocol.OpPeerOffer:
			err = n.serveOffer(conn, req)
		case protocol.OpPeerPush:
			err = n.servePeerPush(conn, req)
		default:
			err = writeFrame(conn, &protocol.Message{ID: req.ID, Op: req.Op, Error: "unknown peer op " + req.Op})
		}
		if err != nil {
			n.log.Debug("peer serve", "remote", conn.RemoteAddr(), "op", req.Op, "err", err)
			return
		}
	}
}

func (n *Node) serveGet(conn net.Conn, req *protocol.Message) error {
	arena := n.store.Arena()
	resp := &protocol.Message{ID: req.ID, Op: req.Op}
	if arena == nil {
		resp.Error = "pool not initialised"
		return writeFrame(conn, resp)
	}
	if req.SlotBytes != arena.SlotBytes() {
		resp.Error = fmt.Sprintf("slot size %d does not match %d", req.SlotBytes, arena.SlotBytes())
		return writeFrame(conn, resp)
	}
	sid := n.session()
	defer n.store.ReleaseSession(sid)
	slots := n.store.Pin(sid, req.Namespace, req.Keys)
	resp.Status = make([]uint8, len(slots))
	var found [][]byte
	for i, s := range slots {
		if s < 0 {
			resp.Status[i] = protocol.FetchMissing
			continue
		}
		resp.Status[i] = protocol.FetchOK
		found = append(found, req.Keys[i])
	}
	if req.Push {
		return n.servePush(conn, req, resp, slots, found, arena, sid)
	}
	if req.RDMA && n.rdma != nil {
		handled, err := n.serveRDMA(conn, req, resp, slots, found, arena, sid)
		if handled || err != nil {
			return err
		}
	}
	if err := writeFrame(conn, resp); err != nil {
		return err
	}
	for _, s := range slots {
		if s < 0 {
			continue
		}
		if _, err := conn.Write(arena.Slot(s)); err != nil {
			return err
		}
		n.BytesOut.Add(arena.SlotBytes())
	}
	n.store.Unpin(sid, req.Namespace, found)
	n.Served.Add(uint64(len(found)))
	if req.Evict && len(found) > 0 {
		n.remove(req.Namespace, found)
	}
	return nil
}

// remove drops keys from the pool and announces the ones removed.
func (n *Node) remove(ns string, keys [][]byte) {
	n.notify.Ordered(func() {
		if removed := n.store.Remove(ns, keys); len(removed) > 0 {
			n.notify.AnnounceRemoved(removed)
		}
	})
}

// serveOffer accepts a request to pull blocks from another daemon. It
// replies at once and pulls in the background.
func (n *Node) serveOffer(c net.Conn, req *protocol.Message) error {
	if err := writeFrame(c, &protocol.Message{ID: req.ID, Op: req.Op}); err != nil {
		return err
	}
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), n.opts.Timeout)
		defer cancel()
		if _, err := n.Fetch(ctx, req.Endpoint, req.Namespace, req.Keys, req.Evict); err != nil {
			n.log.Warn("offered pull failed", "source", req.Endpoint, "err", err)
		}
	}()
	return nil
}

// ---------------------------------------------------------------------------
// Pulling from other daemons
// ---------------------------------------------------------------------------

// Fetch pulls keys from the daemon at endpoint into the local pool. When
// evict is set, the source drops the blocks it sends. It returns a
// protocol.Fetch* status per key.
func (n *Node) Fetch(ctx context.Context, endpoint, ns string, keys [][]byte, evict bool) ([]uint8, error) {
	status := make([]uint8, len(keys))
	if n.isSelf(endpoint) {
		// The router named this node as the source: the blocks are local
		// or gone.
		for i, k := range keys {
			if n.store.Has(ns, k) {
				status[i] = protocol.FetchPresent
			} else {
				status[i] = protocol.FetchMissing
			}
		}
		return status, nil
	}
	arena := n.store.Arena()
	if arena == nil {
		return nil, errors.New("pool not initialised")
	}
	sid := n.session()
	// An RDMA failure with unknown state may leave writes in flight into
	// the reserved slots. Then they are held until the helper that ran
	// them has exited (quarantine), not freed at once.
	unsafe := false
	defer func() {
		if !unsafe {
			n.store.ReleaseSession(sid)
			return
		}
		time.AfterFunc(n.helperWait(), func() { n.store.ReleaseSession(sid) })
	}()

	var slots []int64
	var st []pool.AllocStatus
	var err error
	n.notify.Ordered(func() {
		var evicted []pool.KeyRef
		slots, st, evicted, err = n.store.Alloc(sid, ns, keys)
		if len(evicted) > 0 {
			n.notify.AnnounceRemoved(evicted)
		}
	})
	if err != nil {
		return nil, err
	}
	var want [][]byte
	var wantIdx []int
	for i, s := range st {
		switch s {
		case pool.Present:
			status[i] = protocol.FetchPresent
		case pool.NoSpace:
			status[i] = protocol.FetchNoSpace
		case pool.Allocated:
			want = append(want, keys[i])
			wantIdx = append(wantIdx, i)
		}
	}
	if len(want) == 0 {
		return status, nil
	}

	start := time.Now()
	ok := make([]bool, len(want))
	mode := "tcp"
	if n.rdmaReady(ctx) {
		// Prefer a push: the source writes into our slots. A source that
		// cannot push is read from instead, or sends over TCP.
		mode = "rdma-write"
		err = n.fetchPush(ctx, endpoint, ns, want, wantIdx, slots, status, ok, evict, arena)
		if errors.Is(err, ErrNoRemoteRDMA) && !errors.Is(err, ErrRDMAUnsafe) {
			mode = "rdma-read"
			err = n.fetchRDMA(ctx, endpoint, ns, want, wantIdx, slots, status, ok, evict, arena)
		}
	} else {
		err = n.fetchTCP(ctx, endpoint, ns, want, wantIdx, slots, status, ok, evict, arena)
	}
	if err != nil {
		for j, i := range wantIdx {
			if !ok[j] && status[i] == protocol.FetchOK {
				status[i] = protocol.FetchFailed
			}
		}
	}
	var got uint64
	for _, o := range ok {
		if o {
			got++
		}
	}
	el := time.Since(start)
	n.log.Info("fetched blocks", "source", endpoint, "blocks", got, "of", len(want),
		"bytes", got*arena.SlotBytes(), "mode", mode, "ms", el.Milliseconds(),
		"gbps", float64(got*arena.SlotBytes())*8/el.Seconds()/1e9)
	commitKeys, commitOK := want, ok
	if errors.Is(err, ErrRDMAUnsafe) {
		unsafe = true
		n.log.Warn("rdma fetch state unknown; holding its slots until the transfer has surely stopped",
			"source", endpoint, "hold", n.helperWait(), "err", err)
		// Commit only the blocks that arrived. The rest stay reserved, and
		// the deferred release frees them after the quarantine.
		commitKeys, commitOK = nil, nil
		for j, o := range ok {
			if o {
				commitKeys = append(commitKeys, want[j])
				commitOK = append(commitOK, true)
			}
		}
	}
	var stored []pool.KeyRef
	var cerr error
	n.notify.Ordered(func() {
		stored, cerr = n.store.Commit(sid, ns, commitKeys, commitOK)
		if len(stored) > 0 {
			n.notify.AnnounceStored(ns, stored)
		}
	})
	if cerr != nil && err == nil {
		err = cerr
	}
	n.Fetched.Add(uint64(len(stored)))
	return status, err
}

// fetchTCP pulls want over several TCP connections; one TCP stream does not
// fill a fast NIC. Bytes go straight into the reserved slots.
func (n *Node) fetchTCP(ctx context.Context, endpoint, ns string, want [][]byte, wantIdx []int,
	slots []int64, status []uint8, ok []bool, evict bool, arena *pool.Arena,
) error {
	streams := n.opts.Streams
	if streams > len(want) {
		streams = len(want)
	}
	per := (len(want) + streams - 1) / streams
	errs := make([]error, streams)
	var wg sync.WaitGroup
	for stream := 0; stream < streams; stream++ {
		lo, hi := stream*per, (stream+1)*per
		if hi > len(want) {
			hi = len(want)
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(stream, lo, hi int) {
			defer wg.Done()
			errs[stream] = n.pull(ctx, endpoint, ns, want[lo:hi], evict, arena.SlotBytes(), func(j int, found bool, c net.Conn) error {
				jj := lo + j
				i := wantIdx[jj]
				if !found {
					status[i] = protocol.FetchMissing
					return nil
				}
				if _, err := io.ReadFull(c, arena.Slot(slots[i])); err != nil {
					status[i] = protocol.FetchFailed
					return err
				}
				n.BytesIn.Add(arena.SlotBytes())
				ok[jj] = true
				status[i] = protocol.FetchOK
				return nil
			})
		}(stream, lo, hi)
	}
	wg.Wait()
	return errors.Join(errs...)
}

// pull sends peer_get and calls each(j, found, conn) for every requested
// key in order. each must consume the block's bytes when found is true.
func (n *Node) pull(ctx context.Context, endpoint, ns string, keys [][]byte, evict bool, slotBytes uint64,
	each func(j int, found bool, conn net.Conn) error,
) error {
	conn, err := n.get(ctx, endpoint)
	if err != nil {
		return err
	}
	healthy := false
	defer func() { n.put(endpoint, conn, healthy) }()
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(n.opts.Timeout)
	}
	if err := conn.SetDeadline(deadline); err != nil {
		return err
	}

	req := &protocol.Message{ID: 1, Op: protocol.OpPeerGet, Namespace: ns, Keys: keys, Evict: evict, SlotBytes: slotBytes}
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
	if len(resp.Status) != len(keys) {
		return fmt.Errorf("source %s returned %d statuses for %d keys", endpoint, len(resp.Status), len(keys))
	}
	for j, s := range resp.Status {
		if err := each(j, s == protocol.FetchOK, conn); err != nil {
			return err
		}
	}
	healthy = true
	return nil
}

// Offer asks the daemon at endpoint to pull keys from this daemon.
func (n *Node) Offer(ctx context.Context, endpoint, ns string, keys [][]byte, evict bool) error {
	c, err := n.get(ctx, endpoint)
	if err != nil {
		return err
	}
	healthy := false
	defer func() { n.put(endpoint, c, healthy) }()
	if err := c.SetDeadline(time.Now().Add(n.opts.Timeout)); err != nil {
		return err
	}
	req := &protocol.Message{ID: 1, Op: protocol.OpPeerOffer, Namespace: ns, Keys: keys, Evict: evict, Endpoint: n.opts.Endpoint}
	if err := writeFrame(c, req); err != nil {
		return err
	}
	resp, err := protocol.Read(c)
	if err != nil {
		return err
	}
	healthy = true
	if resp.Error != "" {
		return fmt.Errorf("destination %s: %s", endpoint, resp.Error)
	}
	return nil
}

func (n *Node) get(ctx context.Context, endpoint string) (net.Conn, error) {
	addr := HostPort(endpoint)
	for {
		n.mu.Lock()
		cs := n.idle[addr]
		if len(cs) == 0 {
			n.mu.Unlock()
			break
		}
		c := cs[len(cs)-1]
		n.idle[addr] = cs[:len(cs)-1]
		n.mu.Unlock()
		if idleAlive(c) {
			return c, nil
		}
		// The peer closed it, for example because its daemon restarted.
		c.Close()
	}
	d := net.Dialer{Timeout: n.opts.Timeout}
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, err
	}
	if tc, ok := c.(*net.TCPConn); ok {
		if err := tc.SetNoDelay(true); err != nil {
			n.log.Debug("peer set no delay", "endpoint", endpoint, "err", err)
		}
	}
	return c, nil
}

// idleAlive reports whether an idle connection is still open. An idle
// connection has nothing to read, so a read that returns at once, with
// data, EOF or an error, means the peer closed or broke it. A restarted
// daemon leaves every pooled connection to it in that state.
func idleAlive(c net.Conn) bool {
	if err := c.SetReadDeadline(time.Now().Add(time.Millisecond)); err != nil {
		return false
	}
	var b [1]byte
	_, err := c.Read(b[:])
	var ne net.Error
	if !errors.As(err, &ne) || !ne.Timeout() {
		return false
	}
	return c.SetReadDeadline(time.Time{}) == nil
}

func (n *Node) put(endpoint string, c net.Conn, healthy bool) {
	if err := c.SetDeadline(time.Time{}); err != nil {
		n.log.Debug("peer clear deadline", "endpoint", endpoint, "err", err)
		healthy = false
	}
	addr := HostPort(endpoint)
	n.mu.Lock()
	defer n.mu.Unlock()
	if !healthy || n.closing.Load() || len(n.idle[addr]) >= n.opts.MaxIdleConns {
		c.Close()
		return
	}
	n.idle[addr] = append(n.idle[addr], c)
}

func writeFrame(w io.Writer, m *protocol.Message) error {
	frame, err := protocol.Encode(m)
	if err != nil {
		return err
	}
	_, err = w.Write(frame)
	return err
}
