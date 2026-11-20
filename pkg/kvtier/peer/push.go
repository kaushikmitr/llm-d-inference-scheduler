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
	"context"
	"crypto/subtle"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvtier/protocol"
)

// Reasons a hash is not pushed, in a push response's Failures.
const (
	PushNotFound = "not_found"
)

// PushByHash offers the blocks with these router hashes to the daemon at
// dest, which pulls them. With evict, the source drops what it sent. A
// namespace ending in "*" matches every namespace with that prefix, for
// callers that know the model but not the layout digest. It returns the
// hashes that were not offered, with the reason.
func (n *Node) PushByHash(ctx context.Context, ns string, hashes []uint64, dest string, evict bool) (map[uint64]string, error) {
	namespaces := []string{ns}
	if prefix, ok := strings.CutSuffix(ns, "*"); ok {
		namespaces = n.store.Namespaces(prefix)
	}
	offered := make(map[uint64]struct{}, len(hashes))
	var errs []error
	for _, name := range namespaces {
		found := n.store.Resolve(name, hashes)
		var keys [][]byte
		for _, h := range hashes {
			if ks, ok := found[h]; ok {
				keys = append(keys, ks...)
				offered[h] = struct{}{}
			}
		}
		if len(keys) == 0 {
			continue
		}
		if err := n.Offer(ctx, dest, name, keys, evict); err != nil {
			errs = append(errs, err)
		}
	}
	var failures map[uint64]string
	for _, h := range hashes {
		if _, ok := offered[h]; !ok {
			if failures == nil {
				failures = make(map[uint64]string)
			}
			failures[h] = PushNotFound
		}
	}
	return failures, errors.Join(errs...)
}

// servePeerPush answers the direct call from a router.
func (n *Node) servePeerPush(conn net.Conn, req *protocol.Message) error {
	resp := &protocol.Message{ID: req.ID, Op: req.Op}
	switch {
	case n.opts.ControlToken == "":
		resp.Error = "direct push is disabled: this daemon has no control token"
	case subtle.ConstantTimeCompare([]byte(req.Token), []byte(n.opts.ControlToken)) != 1:
		resp.Error = "unauthorized"
	case req.Namespace == "" || req.Endpoint == "" || len(req.Hashes) == 0:
		resp.Error = "peer_push needs a namespace, a destination endpoint and hashes"
	default:
		ctx, cancel := context.WithTimeout(context.Background(), n.opts.Timeout)
		defer cancel()
		failures, err := n.PushByHash(ctx, req.Namespace, req.Hashes, req.Endpoint, req.Evict)
		resp.Failures = failures
		if err != nil {
			resp.Error = err.Error()
		}
	}
	return writeFrame(conn, resp)
}

// RequestPush is the client side of the direct call: it asks the source
// daemon at source to send the blocks with these router hashes to the daemon
// at dest. It returns the hashes the source did not send, with the reason.
func RequestPush(ctx context.Context, source, token, ns string, hashes []uint64, dest string, evict bool,
	timeout time.Duration,
) (map[uint64]string, error) {
	d := net.Dialer{Timeout: timeout}
	conn, err := d.DialContext(ctx, "tcp", HostPort(source))
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return nil, err
	}
	req := &protocol.Message{
		ID: 1, Op: protocol.OpPeerPush, Token: token, Namespace: ns, Hashes: hashes, Endpoint: dest, Evict: evict,
	}
	if err := writeFrame(conn, req); err != nil {
		return nil, err
	}
	resp, err := protocol.Read(conn)
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return resp.Failures, fmt.Errorf("source %s: %s", source, resp.Error)
	}
	return resp.Failures, nil
}
