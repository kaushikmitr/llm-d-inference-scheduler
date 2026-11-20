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
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/peer"
	"github.com/llm-d/llm-d-router/pkg/kvtier/pool"
)

const testToken = "s3cret"

// stored puts one block on a daemon and returns its key and router hash.
func stored(t *testing.T, d daemon, ns string, hash uint64, data string) ([]byte, uint64) {
	t.Helper()
	key := vllmKey(hash)
	dial(t, d.sock, ns).storeBlock(key, data)
	h, ok := pool.RouterHash(key)
	if !ok {
		t.Fatal("no router hash")
	}
	return key, h
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestDirectPushMovesBlocksWithoutAnEngineRequest(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, "ns", 0xfeed, "direct")
	cb := dial(t, dst.sock, "ns")

	failures, err := peer.RequestPush(t.Context(), src.endpoint, testToken, "ns", []uint64{h, 99}, dst.endpoint, false, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if failures[99] != peer.PushNotFound || len(failures) != 1 {
		t.Fatalf("failures %v, want only hash 99 not found", failures)
	}
	waitFor(t, "the block on the destination", func() bool { return dst.store.Has("ns", key) })
	if got := cb.read(key, 6); got != "direct" {
		t.Fatalf("destination read %q", got)
	}
	if !src.store.Has("ns", key) {
		t.Fatal("a push without evict removed the source copy")
	}
}

func TestDirectPushWithEvictMoves(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, "ns", 0xbeef, "moved")
	dial(t, dst.sock, "ns")

	if _, err := peer.RequestPush(t.Context(), src.endpoint, testToken, "ns", []uint64{h}, dst.endpoint, true, 5*time.Second); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the block on the destination", func() bool { return dst.store.Has("ns", key) })
	waitFor(t, "the source to drop its copy", func() bool { return !src.store.Has("ns", key) })
}

func TestDirectPushMatchesNamespacesByPrefix(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, "model-x:abc123", 0xcafe, "byprefix")
	other, _ := stored(t, src, "model-y:abc123", 0xcafe, "other")
	dial(t, dst.sock, "model-x:abc123")

	if _, err := peer.RequestPush(t.Context(), src.endpoint, testToken, "model-x:*", []uint64{h}, dst.endpoint, false, 5*time.Second); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the block on the destination", func() bool { return dst.store.Has("model-x:abc123", key) })
	time.Sleep(100 * time.Millisecond)
	if dst.store.Has("model-y:abc123", other) {
		t.Fatal("the prefix matched another model's namespace")
	}
}

func TestDirectPushNeedsTheToken(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, "ns", 0xf00d, "guarded")
	dial(t, dst.sock, "ns")

	for _, token := range []string{"", "wrong"} {
		_, err := peer.RequestPush(t.Context(), src.endpoint, token, "ns", []uint64{h}, dst.endpoint, true, 5*time.Second)
		if err == nil || !strings.Contains(err.Error(), "unauthorized") {
			t.Fatalf("token %q: err %v, want unauthorized", token, err)
		}
	}
	time.Sleep(100 * time.Millisecond)
	if dst.store.Has("ns", key) || !src.store.Has("ns", key) {
		t.Fatal("an unauthorized push moved a block")
	}
}

func TestDirectPushIsOffWithoutAToken(t *testing.T) {
	src, dst := startDaemon(t), startDaemon(t)
	_, h := stored(t, src, "ns", 0xd00d, "off")

	_, err := peer.RequestPush(t.Context(), src.endpoint, "anything", "ns", []uint64{h}, dst.endpoint, false, 5*time.Second)
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("err %v, want direct push disabled", err)
	}
}
