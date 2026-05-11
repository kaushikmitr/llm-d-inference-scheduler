/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nohitlru

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// makeEndpoint creates a test endpoint with optional attributes for testing.
func makeEndpoint(name string, prefixMatch int, ttft float64, tokens int64) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if prefixMatch >= 0 {
		ep.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(prefixMatch, 100, 16))
	}
	if ttft >= 0 {
		ep.Put(attrlatency.LatencyPredictionInfoKey, attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, ttft, 0, 0))
	}
	if tokens >= 0 {
		ep.Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{Tokens: tokens})
	}
	return ep
}

// mockHandle implements fwkplugin.Handle for testing.
type mockHandle struct {
	fwkplugin.Handle
	ctx context.Context
}

func (m *mockHandle) Context() context.Context { return m.ctx }

func TestFactory_ValidConfig(t *testing.T) {
	handle := &mockHandle{ctx: context.Background()}
	plugin, err := Factory("test", nil, handle)
	assert.NoError(t, err)
	assert.NotNil(t, plugin)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
}

func TestFilter_WarmRequestPassedThrough(t *testing.T) {
	handle := &mockHandle{ctx: context.Background()}
	plugin, _ := Factory("test", nil, handle)
	p := plugin.(*Plugin)

	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 50, 10, 0), // warm
		makeEndpoint("b", 0, 20, 0),  // cold
	}
	req := &fwksched.InferenceRequest{RequestID: "req-1"}
	result := p.Filter(context.Background(), nil, req, endpoints)

	assert.Equal(t, 2, len(result), "warm request should not be filtered")
}

func TestFilter_ColdRequestLRU(t *testing.T) {
	handle := &mockHandle{ctx: context.Background()}
	plugin, _ := Factory("test", nil, handle)
	p := plugin.(*Plugin)

	// Populate LRU cache manually
	p.lruCache.Add("default/a", struct{}{}) // oldest
	p.lruCache.Add("default/b", struct{}{}) // newest

	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 0, 10, 0), // used (oldest)
		makeEndpoint("b", 0, 20, 0), // used (newest)
	}
	req := &fwksched.InferenceRequest{RequestID: "req-2"}
	result := p.Filter(context.Background(), nil, req, endpoints)

	assert.Equal(t, 1, len(result), "should narrow down to the least recently used")
	assert.Equal(t, "default/a", result[0].GetMetadata().NamespacedName.String(), "oldest used should be selected")
}

func TestFilter_ColdRequestLoadGate(t *testing.T) {
	handle := &mockHandle{ctx: context.Background()}
	plugin, _ := Factory("test", []byte(`{"maxTTFTPenaltyMs": 100}`), handle)
	p := plugin.(*Plugin)

	// Force LRU state
	p.lruCache.Add("default/a", struct{}{}) // oldest but very slow
	p.lruCache.Add("default/b", struct{}{}) // newest but fast

	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 0, 500, 0), // diff is 450 > penalty of 100
		makeEndpoint("b", 0, 50, 0),
	}
	req := &fwksched.InferenceRequest{RequestID: "req-3"}
	result := p.Filter(context.Background(), nil, req, endpoints)

	// Since 'a' is LRU but too slow, it should break the lock and return all
	assert.Equal(t, 2, len(result), "load gate should break the LRU lock and return all")
}
