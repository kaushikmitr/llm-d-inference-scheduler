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

package peer //nolint:testpackage // get and put are unexported

import (
	"io"
	"log/slog"
	"net"
	"testing"
	"time"
)

// A restarted peer leaves pooled connections to it closed. get must skip
// them and dial a fresh one, not hand out a dead connection.
func TestGetSkipsIdleConnectionsThePeerClosed(t *testing.T) {
	var lc net.ListenConfig
	ln, err := lc.Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	accepted := make(chan net.Conn, 4)
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			accepted <- c
		}
	}()
	node := &Node{
		log:   slog.New(slog.NewTextHandler(io.Discard, nil)),
		opts:  Options{Timeout: time.Second, MaxIdleConns: 4},
		idle:  map[string][]net.Conn{},
		conns: map[net.Conn]struct{}{},
	}
	endpoint := "tcp://" + ln.Addr().String()

	stale, err := node.get(t.Context(), endpoint)
	if err != nil {
		t.Fatal(err)
	}
	node.put(endpoint, stale, true)
	// The peer closes its end, as a restarting daemon does.
	if err := (<-accepted).Close(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)

	fresh, err := node.get(t.Context(), endpoint)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = fresh.Close() })
	if fresh == stale {
		t.Fatal("get returned the connection the peer closed")
	}
	select {
	case c := <-accepted:
		t.Cleanup(func() { _ = c.Close() })
	case <-time.After(time.Second):
		t.Fatal("get did not dial a fresh connection")
	}

	// A live idle connection is reused.
	node.put(endpoint, fresh, true)
	again, err := node.get(t.Context(), endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if again != fresh {
		t.Fatal("get did not reuse a live idle connection")
	}
}
