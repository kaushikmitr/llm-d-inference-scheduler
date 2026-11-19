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
	"encoding/binary"
	"io"
	"log/slog"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/peer"
	"github.com/llm-d/llm-d-router/pkg/kvtier/policy"
	"github.com/llm-d/llm-d-router/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
	"github.com/llm-d/llm-d-router/pkg/kvtier/server"
)

type daemon struct {
	sock     string
	endpoint string
	store    *pool.Store
}

// startDaemon starts a daemon with a pool of four slots and peer transfer
// enabled.
func startDaemon(t *testing.T) daemon {
	t.Helper()
	const slots = 4
	store := pool.NewStore(pool.ArenaOptions{CapacityBytes: slots * slotBytes}, func() policy.Policy { return policy.NewLRU() })
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := server.New(store, log)
	node := peer.New(store, srv, log, peer.Options{ListenAddr: "127.0.0.1:0", Timeout: 5 * time.Second})
	if err := node.Listen(); err != nil {
		t.Fatal(err)
	}
	srv.SetPeer(node)
	path := filepath.Join(t.TempDir(), "d.sock")
	if err := srv.Listen(path); err != nil {
		t.Fatal(err)
	}
	serve(t, srv, node.Close)
	return daemon{sock: path, endpoint: node.Endpoint(), store: store}
}

// vllmKey builds a vLLM-style offload key: an 8-byte block hash and a
// 4-byte group index.
func vllmKey(hash uint64) []byte {
	k := make([]byte, 12)
	binary.BigEndian.PutUint64(k, hash)
	return k
}

func (c *client) storeBlock(key []byte, data string) {
	c.t.Helper()
	r := c.call(&protocol.Message{Op: protocol.OpAlloc, Keys: [][]byte{key}})
	copy(c.mem[r.Slots[0]*slotBytes:], data)
	c.call(&protocol.Message{Op: protocol.OpCommit, Keys: [][]byte{key}, OK: []bool{true}})
}

func (c *client) read(key []byte, n int) string {
	c.t.Helper()
	r := c.call(&protocol.Message{Op: protocol.OpPin, Keys: [][]byte{key}})
	if r.Slots[0] < 0 {
		return ""
	}
	defer c.call(&protocol.Message{Op: protocol.OpUnpin, Keys: [][]byte{key}})
	off := r.Slots[0] * slotBytes
	return string(c.mem[off : off+int64(n)])
}

func TestFetchFromPeer(t *testing.T) {
	a, b := startDaemon(t), startDaemon(t)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k1, k2 := vllmKey(1), vllmKey(2)
	ca.storeBlock(k1, "from-a")

	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k1, k2}})
	if r.Status[0] != protocol.FetchOK || r.Status[1] != protocol.FetchMissing {
		t.Fatalf("status %v", r.Status)
	}
	if got := cb.read(k1, 6); got != "from-a" {
		t.Fatalf("b read %q", got)
	}
	if !a.store.Has("ns", k1) {
		t.Fatal("plain fetch removed the source copy")
	}
	r = cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k1}})
	if r.Status[0] != protocol.FetchPresent {
		t.Fatalf("second fetch status %v, want present", r.Status)
	}
}

func TestFetchNamespacesDoNotMix(t *testing.T) {
	a, b := startDaemon(t), startDaemon(t)
	dial(t, a.sock, "model-x").storeBlock(vllmKey(1), "x")
	cb := dial(t, b.sock, "model-y")
	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{vllmKey(1)}})
	if r.Status[0] != protocol.FetchMissing {
		t.Fatalf("status %v, want missing across namespaces", r.Status)
	}
}

func TestEvictOnFetch(t *testing.T) {
	a, b := startDaemon(t), startDaemon(t)
	ca, cb := dial(t, a.sock, "ns"), dial(t, b.sock, "ns")
	k := vllmKey(7)
	ca.storeBlock(k, "moved")
	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{k}, Evict: true})
	if r.Status[0] != protocol.FetchOK {
		t.Fatalf("status %v", r.Status)
	}
	if a.store.Has("ns", k) {
		t.Fatal("source kept the block after evict-on-fetch")
	}
	if ev := ca.event(); len(ev.Removed) != 1 {
		t.Fatalf("source client event %+v, want a removal", ev)
	}
	if got := cb.read(k, 5); got != "moved" {
		t.Fatalf("b read %q", got)
	}
}

func TestPushMovesBlocksByRouterHash(t *testing.T) {
	da, db := startDaemon(t), startDaemon(t)
	ca := dial(t, da.sock, "ns")
	cb := dial(t, db.sock, "ns")
	key := vllmKey(0xdeadbeef)
	ca.storeBlock(key, "pushed")
	h, ok := pool.RouterHash(key)
	if !ok {
		t.Fatal("no router hash")
	}

	r := ca.call(&protocol.Message{Op: protocol.OpPush, Endpoint: db.endpoint, Hashes: []uint64{h, 99}, Evict: true})
	if r.Failures[99] != "not_found" || len(r.Failures) != 1 {
		t.Fatalf("failures %v", r.Failures)
	}
	if ev := cb.event(); len(ev.Stored) != 1 {
		t.Fatalf("destination event %+v, want stored", ev)
	}
	if got := cb.read(key, 6); got != "pushed" {
		t.Fatalf("db read %q", got)
	}
	deadline := time.Now().Add(5 * time.Second)
	for da.store.Has("ns", key) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if da.store.Has("ns", key) {
		t.Fatal("move kept the source copy")
	}
}

func TestFetchFromSelf(t *testing.T) {
	a := startDaemon(t)
	ca := dial(t, a.sock, "ns")
	ca.storeBlock(vllmKey(1), "x")
	r := ca.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{vllmKey(1), vllmKey(2)}})
	if r.Status[0] != protocol.FetchPresent || r.Status[1] != protocol.FetchMissing {
		t.Fatalf("status %v", r.Status)
	}
}

func TestFetchSlotSizeMismatch(t *testing.T) {
	a, b := startDaemon(t), startDaemon(t)
	// Daemon a's arena uses a different slot size.
	if _, err := a.store.EnsureArena(slotBytes * 2); err != nil {
		t.Fatal(err)
	}
	cb := dial(t, b.sock, "ns")
	r := cb.call(&protocol.Message{Op: protocol.OpFetch, Endpoint: a.endpoint, Keys: [][]byte{vllmKey(1)}})
	if r.Status[0] != protocol.FetchFailed {
		t.Fatalf("status %v, want failed", r.Status)
	}
	_, _, ev, err := b.store.Alloc(1, "ns", [][]byte{vllmKey(1)})
	if err != nil {
		t.Fatal(err)
	}
	if len(ev) != 0 {
		t.Fatal("failed fetch left the slot occupied")
	}
}

func TestRouterHash(t *testing.T) {
	k := vllmKey(0x0102030405060708)
	if h, _ := pool.RouterHash(k); h != 0x0102030405060708 {
		t.Fatalf("hash %x", h)
	}
	// A 32-byte sha256 block hash: the low 64 bits are its last 8 bytes.
	long := make([]byte, 36)
	binary.BigEndian.PutUint64(long[24:32], 42)
	if h, _ := pool.RouterHash(long); h != 42 {
		t.Fatalf("hash %d", h)
	}
}
