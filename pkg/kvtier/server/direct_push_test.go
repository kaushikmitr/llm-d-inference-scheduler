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
	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
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

const testModel = "model-x"

// modelNS is a namespace of testModel, as the vLLM shim names it.
const modelNS = testModel + ":abc123"

func sendPrefetch(t *testing.T, src daemon, token, dest string, hashes []uint64, move bool) (*peer.HintResult, error) {
	t.Helper()
	return peer.SendHints(t.Context(), src.endpoint, token, testModel, peer.PrefetchHint("m-1", dest, hashes, move), 5*time.Second)
}

func TestDirectHintPrefetchesWithoutAnEngineRequest(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, modelNS, 0xfeed, "direct")
	cb := dial(t, dst.sock, modelNS)

	res, err := sendPrefetch(t, src, testToken, dst.endpoint, []uint64{h, 99}, false)
	if err != nil {
		t.Fatal(err)
	}
	if res.Failures[99] != peer.PushNotFound || len(res.Failures) != 1 || len(res.Rejected) != 0 {
		t.Fatalf("failures %v rejected %v, want only hash 99 not found", res.Failures, res.Rejected)
	}
	waitFor(t, "the block on the destination", func() bool { return dst.store.Has(modelNS, key) })
	if got := cb.read(key, 6); got != "direct" {
		t.Fatalf("destination read %q", got)
	}
	if !src.store.Has(modelNS, key) {
		t.Fatal("a prefetch removed the source copy")
	}
}

func TestDirectHintMoveDropsTheSourceCopy(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, modelNS, 0xbeef, "moved")
	dial(t, dst.sock, modelNS)

	if _, err := sendPrefetch(t, src, testToken, dst.endpoint, []uint64{h}, true); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the block on the destination", func() bool { return dst.store.Has(modelNS, key) })
	waitFor(t, "the source to drop its copy", func() bool { return !src.store.Has(modelNS, key) })
}

func TestDirectHintAddressesOneModel(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, modelNS, 0xcafe, "mine")
	other, _ := stored(t, src, "model-y:abc123", 0xcafe, "other")
	tpuKey, _ := stored(t, src, "tpu:"+testModel+":def456", 0xcafe, "ontpu")
	dial(t, dst.sock, modelNS)

	if _, err := sendPrefetch(t, src, testToken, dst.endpoint, []uint64{h}, false); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the vLLM-shim block", func() bool { return dst.store.Has(modelNS, key) })
	waitFor(t, "the TPU-shim block", func() bool { return dst.store.Has("tpu:"+testModel+":def456", tpuKey) })
	time.Sleep(100 * time.Millisecond)
	if dst.store.Has("model-y:abc123", other) {
		t.Fatal("the hint moved another model's block")
	}
}

func TestDirectHintNeedsTheToken(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, modelNS, 0xf00d, "guarded")
	dial(t, dst.sock, modelNS)

	for _, token := range []string{"", "wrong"} {
		_, err := sendPrefetch(t, src, token, dst.endpoint, []uint64{h}, true)
		if err == nil || !strings.Contains(err.Error(), "unauthorized") {
			t.Fatalf("token %q: err %v, want unauthorized", token, err)
		}
	}
	time.Sleep(100 * time.Millisecond)
	if dst.store.Has(modelNS, key) || !src.store.Has(modelNS, key) {
		t.Fatal("an unauthorized hint moved a block")
	}
}

func TestDirectHintIsOffWithoutAToken(t *testing.T) {
	src, dst := startDaemon(t), startDaemon(t)
	_, h := stored(t, src, modelNS, 0xd00d, "off")

	_, err := sendPrefetch(t, src, "anything", dst.endpoint, []uint64{h}, false)
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Fatalf("err %v, want direct hints disabled", err)
	}
}

func TestDirectHintRejectsWhatItCannotRun(t *testing.T) {
	src, dst := startDaemonWithToken(t, testToken), startDaemon(t)
	key, h := stored(t, src, modelNS, 0xabcd, "kept")
	dial(t, dst.sock, modelNS)
	send := func(env *protocol.HintsEnvelope) (*peer.HintResult, error) {
		return peer.SendHints(t.Context(), src.endpoint, testToken, testModel, env, 5*time.Second)
	}

	// Another protocol version: the whole envelope is refused.
	env := peer.PrefetchHint("m-1", dst.endpoint, []uint64{h}, false)
	env.ProtocolVersion = "0.2"
	if _, err := send(env); err == nil || !strings.Contains(err.Error(), "protocol_version") {
		t.Fatalf("err %v, want an unsupported protocol_version", err)
	}

	// Actions the daemon cannot run are rejected one by one; a good action
	// in the same envelope still runs.
	good := peer.PrefetchHint("m-2", dst.endpoint, []uint64{h}, false).Actions[0]
	good.ActionID = "good"
	fetch := good
	fetch.ActionID, fetch.ActionType = "fetch", "kv.fetch"
	old := good
	old.ActionID, old.ActionVersion = "old", "1.1"
	noDest := good
	noDest.ActionID, noDest.Payload = "no-dest", map[string]any{protocol.HintBlockHashesField: []any{h}}
	env = &protocol.HintsEnvelope{
		ProtocolVersion: protocol.HintProtocolVersion, MessageID: "m-2",
		Actions: []protocol.HintAction{fetch, old, noDest, good},
	}
	res, err := send(env)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Rejected) != 3 || res.Rejected["fetch"] != "unsupported_action_type" ||
		!strings.Contains(res.Rejected["old"], "action_version") || res.Rejected["no-dest"] != "invalid router hint" {
		t.Fatalf("rejected %v", res.Rejected)
	}
	waitFor(t, "the good action's block", func() bool { return dst.store.Has(modelNS, key) })
}
