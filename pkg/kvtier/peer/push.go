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

// namespacesFor returns the namespaces a direct call addresses. An explicit
// namespace names one namespace, or a prefix when it ends in "*". A model
// name matches every layout of that model, for the vLLM and the TPU shims.
func (n *Node) namespacesFor(ns, model string) []string {
	if ns != "" {
		if prefix, ok := strings.CutSuffix(ns, "*"); ok {
			return n.store.Namespaces(prefix)
		}
		return []string{ns}
	}
	out := n.store.Namespaces(model + ":")
	return append(out, n.store.Namespaces("tpu:"+model+":")...)
}

// hintHashes reads the block_hashes of a hint payload. Hashes are 64-bit
// integers; any other value makes the action invalid.
func hintHashes(payload map[string]any) ([]uint64, bool) {
	raw, ok := payload[protocol.HintBlockHashesField].([]any)
	if !ok || len(raw) == 0 {
		return nil, false
	}
	out := make([]uint64, 0, len(raw))
	for _, v := range raw {
		switch h := v.(type) {
		case uint64:
			out = append(out, h)
		case int64:
			if h < 0 {
				return nil, false
			}
			out = append(out, uint64(h))
		case uint32:
			out = append(out, uint64(h))
		case int32:
			if h < 0 {
				return nil, false
			}
			out = append(out, uint64(h))
		case uint16:
			out = append(out, uint64(h))
		case uint8:
			out = append(out, uint64(h))
		case int8:
			if h < 0 {
				return nil, false
			}
			out = append(out, uint64(h))
		case int16:
			if h < 0 {
				return nil, false
			}
			out = append(out, uint64(h))
		default:
			return nil, false
		}
	}
	return out, true
}

// runHints carries out the prefetch and move actions of an envelope. It
// returns the hashes the source did not hold and the actions it did not run.
func (n *Node) runHints(ctx context.Context, namespaces []string, env *protocol.HintsEnvelope) (
	failures map[uint64]string, rejected map[string]string, err error,
) {
	reject := func(id, reason string) {
		if rejected == nil {
			rejected = make(map[string]string)
		}
		rejected[id] = reason
	}
	var errs []error
	for i := range env.Actions {
		act := &env.Actions[i]
		switch {
		case act.ActionType != protocol.HintPrefetch && act.ActionType != protocol.HintMove:
			reject(act.ActionID, "unsupported_action_type")
			continue
		case act.ActionVersion != protocol.HintActionVersion:
			reject(act.ActionID, fmt.Sprintf("unsupported action_version %q; supported %q",
				act.ActionVersion, protocol.HintActionVersion))
			continue
		}
		dest, ok := act.Payload[protocol.HintDestinationField].(string)
		hashes, hashesOK := hintHashes(act.Payload)
		if !ok || dest == "" || !hashesOK {
			reject(act.ActionID, "invalid router hint")
			continue
		}
		evict := act.ActionType == protocol.HintMove
		offered := make(map[uint64]struct{}, len(hashes))
		for _, ns := range namespaces {
			missing, perr := n.PushByHash(ctx, ns, hashes, dest, evict)
			if perr != nil {
				errs = append(errs, perr)
			}
			for _, h := range hashes {
				if _, miss := missing[h]; !miss {
					offered[h] = struct{}{}
				}
			}
		}
		for _, h := range hashes {
			if _, ok := offered[h]; !ok {
				if failures == nil {
					failures = make(map[uint64]string)
				}
				failures[h] = PushNotFound
			}
		}
	}
	return failures, rejected, errors.Join(errs...)
}

// servePeerHints answers the direct call from a router.
func (n *Node) servePeerHints(conn net.Conn, req *protocol.Message) error {
	resp := &protocol.Message{ID: req.ID, Op: req.Op}
	switch {
	case n.opts.ControlToken == "":
		resp.Error = "direct hints are disabled: this daemon has no control token"
	case subtle.ConstantTimeCompare([]byte(req.Token), []byte(n.opts.ControlToken)) != 1:
		resp.Error = "unauthorized"
	case req.Hints == nil || (req.Model == "" && req.Namespace == ""):
		resp.Error = "peer_hints needs a hint envelope and a model"
	case req.Hints.ProtocolVersion != protocol.HintProtocolVersion:
		resp.Error = fmt.Sprintf("unsupported protocol_version %q; supported %q",
			req.Hints.ProtocolVersion, protocol.HintProtocolVersion)
	default:
		ctx, cancel := context.WithTimeout(context.Background(), n.opts.Timeout)
		defer cancel()
		var err error
		resp.Failures, resp.Rejected, err = n.runHints(ctx, n.namespacesFor(req.Namespace, req.Model), req.Hints)
		if err != nil {
			resp.Error = err.Error()
		}
	}
	return writeFrame(conn, resp)
}

// PrefetchHint builds the envelope for one kv.prefetch action, or one kv.move
// action when move is set: send the blocks with these hashes to dest.
func PrefetchHint(messageID, dest string, hashes []uint64, move bool) *protocol.HintsEnvelope {
	actionType := protocol.HintPrefetch
	if move {
		actionType = protocol.HintMove
	}
	list := make([]any, len(hashes))
	for i, h := range hashes {
		list[i] = h
	}
	return &protocol.HintsEnvelope{
		ProtocolVersion: protocol.HintProtocolVersion,
		MessageID:       messageID,
		Actions: []protocol.HintAction{{
			ActionID: "a-0", ActionType: actionType, ActionVersion: protocol.HintActionVersion,
			Payload: map[string]any{protocol.HintDestinationField: dest, protocol.HintBlockHashesField: list},
		}},
	}
}

// HintResult is a daemon's answer to a direct call.
type HintResult struct {
	// Failures lists the hashes the source did not hold.
	Failures map[uint64]string
	// Rejected lists the actions the daemon did not run, by action ID.
	Rejected map[string]string
}

// SendHints is the client side of the direct call: it sends a hint envelope
// for model to the daemon at source.
func SendHints(ctx context.Context, source, token, model string, env *protocol.HintsEnvelope,
	timeout time.Duration,
) (*HintResult, error) {
	d := net.Dialer{Timeout: timeout}
	conn, err := d.DialContext(ctx, "tcp", HostPort(source))
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(timeout)); err != nil {
		return nil, err
	}
	req := &protocol.Message{ID: 1, Op: protocol.OpPeerHints, Token: token, Model: model, Hints: env}
	if err := writeFrame(conn, req); err != nil {
		return nil, err
	}
	resp, err := protocol.Read(conn)
	if err != nil {
		return nil, err
	}
	res := &HintResult{Failures: resp.Failures, Rejected: resp.Rejected}
	if resp.Error != "" {
		return res, fmt.Errorf("daemon %s: %s", source, resp.Error)
	}
	return res, nil
}
