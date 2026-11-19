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

package server_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/peer"
	"github.com/llm-d/llm-d-router/pkg/kvtier/policy"
	"github.com/llm-d/llm-d-router/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
	"github.com/llm-d/llm-d-router/pkg/kvtier/server"
)

// fakeFabric stands in for NIXL between in-process helpers. A helper's
// metadata is its name, its base address is 0, and a remote address is a
// byte offset into the peer's pool.
type fakeFabric struct {
	mu     sync.Mutex
	stores map[string]*pool.Store
}

type fakeHelper struct {
	name   string
	store  *pool.Store
	fabric *fakeFabric
	writes atomic.Int64
	reads  atomic.Int64
	// unsafeWrite makes writes fail as if transfers were left running.
	unsafeWrite atomic.Bool
}

func (f *fakeFabric) arena(name string) (*pool.Arena, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	st := f.stores[name]
	if st == nil || st.Arena() == nil {
		return nil, errors.New("unknown peer " + name)
	}
	return st.Arena(), nil
}

type fakeRequest struct {
	Op       string     `json:"op"`
	PeerMeta []byte     `json:"peer_metadata"`
	Pairs    [][2]int64 `json:"pairs"`
}

func (h *fakeHelper) handle(req *fakeRequest) any {
	local := h.store.Arena()
	if local == nil {
		return map[string]string{"error": "no pool"}
	}
	sb := int64(local.SlotBytes()) // #nosec G115 -- test slot size
	op, peerMeta, pairs := req.Op, req.PeerMeta, req.Pairs
	switch op {
	case "info":
		return map[string]any{"name": h.name, "metadata": []byte(h.name), "base": 0, "slot_bytes": local.SlotBytes()}
	case "write":
		if h.unsafeWrite.Load() {
			return map[string]any{"error": "timed out with transfers running", "unsafe": true}
		}
		dst, err := h.fabric.arena(string(peerMeta))
		if err != nil {
			return map[string]string{"error": err.Error()}
		}
		for _, p := range pairs {
			copy(dst.Slot(p[1]/sb), local.Slot(p[0]))
		}
		h.writes.Add(1)
	case "read":
		src, err := h.fabric.arena(string(peerMeta))
		if err != nil {
			return map[string]string{"error": err.Error()}
		}
		for _, p := range pairs {
			copy(local.Slot(p[0]), src.Slot(p[1]))
		}
		h.reads.Add(1)
	default:
		return map[string]string{"error": "unknown op " + op}
	}
	return map[string]string{}
}

func (h *fakeHelper) serve(t *testing.T, path string) {
	t.Helper()
	var lc net.ListenConfig
	ln, err := lc.Listen(t.Context(), "unix", path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer func() { _ = conn.Close() }()
				rd := bufio.NewReader(conn)
				for {
					line, err := rd.ReadBytes('\n')
					if err != nil {
						return
					}
					req := &fakeRequest{}
					if err := json.Unmarshal(line, req); err != nil {
						return
					}
					out, err := json.Marshal(h.handle(req))
					if err != nil {
						return
					}
					if _, err := conn.Write(append(out, '\n')); err != nil {
						return
					}
				}
			}()
		}
	}()
}

// startRDMADaemon starts a daemon like startDaemon. With withHelper set, it
// also gets a fake RDMA helper on fabric.
func startRDMADaemon(t *testing.T, fabric *fakeFabric, name string, withHelper bool) (daemon, *fakeHelper) {
	t.Helper()
	const slots = 4
	store := pool.NewStore(pool.ArenaOptions{CapacityBytes: slots * slotBytes}, func() policy.Policy { return policy.NewLRU() })
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := server.New(store, log)
	opts := peer.Options{ListenAddr: "127.0.0.1:0", Timeout: 5 * time.Second}
	var h *fakeHelper
	if withHelper {
		h = &fakeHelper{name: name, store: store, fabric: fabric}
		fabric.mu.Lock()
		fabric.stores[name] = store
		fabric.mu.Unlock()
		opts.RDMAHelperSocket = filepath.Join(t.TempDir(), "rdma.sock")
		h.serve(t, opts.RDMAHelperSocket)
	}
	node := peer.New(store, srv, log, opts)
	if err := node.Listen(); err != nil {
		t.Fatal(err)
	}
	srv.SetPeer(node)
	path := filepath.Join(t.TempDir(), "d.sock")
	if err := srv.Listen(path); err != nil {
		t.Fatal(err)
	}
	serve(t, srv, node.Close)
	return daemon{sock: path, endpoint: node.Endpoint(), store: store}, h
}

func TestFetchUsesRDMAWrite(t *testing.T) {
	fabric := &fakeFabric{stores: map[string]*pool.Store{}}
	a, ha := startRDMADaemon(t, fabric, "a", true)
	b, hb := startRDMADaemon(t, fabric, "b", true)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k1, k2 := vllmKey(1), vllmKey(2)
	ca.storeBlock(k1, "pushed")

	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k1, k2}})
	if r.Status[0] != protocol.FetchOK || r.Status[1] != protocol.FetchMissing {
		t.Fatalf("status %v", r.Status)
	}
	if got := cb.read(k1, 6); got != "pushed" {
		t.Fatalf("b read %q", got)
	}
	if ha.writes.Load() != 1 || hb.reads.Load() != 0 {
		t.Fatalf("source writes %d, destination reads %d; want 1 and 0", ha.writes.Load(), hb.reads.Load())
	}
	if !a.store.Has("ns", k1) {
		t.Fatal("plain fetch removed the source copy")
	}
}

func TestFetchRDMAWriteEvicts(t *testing.T) {
	fabric := &fakeFabric{stores: map[string]*pool.Store{}}
	a, _ := startRDMADaemon(t, fabric, "a", true)
	b, _ := startRDMADaemon(t, fabric, "b", true)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k := vllmKey(1)
	ca.storeBlock(k, "moved")

	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k}, Evict: true})
	if r.Status[0] != protocol.FetchOK {
		t.Fatalf("status %v", r.Status)
	}
	if got := cb.read(k, 5); got != "moved" {
		t.Fatalf("b read %q", got)
	}
	if a.store.Has("ns", k) {
		t.Fatal("evicting fetch kept the source copy")
	}
}

func TestFetchFallsBackWhenSourceHasNoRDMA(t *testing.T) {
	fabric := &fakeFabric{stores: map[string]*pool.Store{}}
	a, _ := startRDMADaemon(t, fabric, "a", false)
	b, hb := startRDMADaemon(t, fabric, "b", true)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k := vllmKey(1)
	ca.storeBlock(k, "over-tcp")

	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k}})
	if r.Status[0] != protocol.FetchOK {
		t.Fatalf("status %v", r.Status)
	}
	if got := cb.read(k, 8); got != "over-tcp" {
		t.Fatalf("b read %q", got)
	}
	if hb.reads.Load() != 0 {
		t.Fatalf("destination ran %d RDMA reads against a source with no helper", hb.reads.Load())
	}
}

func TestUnsafePushHoldsTheDestinationSlots(t *testing.T) {
	fabric := &fakeFabric{stores: map[string]*pool.Store{}}
	a, ha := startRDMADaemon(t, fabric, "a", true)
	b, _ := startRDMADaemon(t, fabric, "b", true)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k := vllmKey(1)
	ca.storeBlock(k, "late")
	ha.unsafeWrite.Store(true)

	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k}})
	if r.Status[0] != protocol.FetchFailed {
		t.Fatalf("status %v, want failed", r.Status)
	}
	if b.store.Has("ns", k) {
		t.Fatal("a block with unknown transfer state was committed")
	}
	// Writes may still land in the slot b reserved, so b holds it: only
	// three of its four slots can be allocated now.
	keys := [][]byte{vllmKey(10), vllmKey(11), vllmKey(12), vllmKey(13)}
	got := cb.call(&protocol.Message{Op: protocol.OpAlloc, Keys: keys})
	allocated := 0
	for _, st := range got.Status {
		if st == uint8(pool.Allocated) {
			allocated++
		}
	}
	if allocated != 3 {
		t.Fatalf("allocated %d of 4 slots, want 3 while one is held", allocated)
	}
}
