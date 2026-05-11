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
	"encoding/json"
	"fmt"
	"math"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	PluginType            = "no-hit-lru-filter"
	defaultLRUSize        = 1024
	defaultPrefillProfile = "prefill"
)

var _ fwksched.Filter = &Plugin{}
var _ requestcontrol.PreRequest = &Plugin{}

type Config struct {
	LRUSize                  int     `json:"lruSize,omitempty"`
	MaxTTFTPenaltyMs         float64 `json:"maxTTFTPenaltyMs,omitempty"`
	MaxTokensInFlightPenalty int64   `json:"maxTokensInFlightPenalty,omitempty"`
}

var DefaultConfig = Config{
	LRUSize:                  defaultLRUSize,
	MaxTTFTPenaltyMs:         5000,
	MaxTokensInFlightPenalty: 0, // 0 means disabled
}

type coldRequestState struct {
	isCold bool
}

func (c *coldRequestState) Clone() fwkplugin.StateData {
	return &coldRequestState{isCold: c.isCold}
}

type Plugin struct {
	typedName   fwkplugin.TypedName
	config      Config
	lruCache    *lru.Cache[string, struct{}]
	pluginState *fwkplugin.PluginState
}

func Factory(name string, rawParameters json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for %s: %w", PluginType, err)
		}
	}

	if config.LRUSize <= 0 {
		config.LRUSize = defaultLRUSize
	}

	lruCache, err := lru.New[string, struct{}](config.LRUSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create LRU cache: %w", err)
	}

	return &Plugin{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		config:      config,
		lruCache:    lruCache,
		pluginState: fwkplugin.NewPluginState(handle.Context()),
	}, nil
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *Plugin) Consumes() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
		attrprefix.PrefixCacheMatchInfoKey:   attrprefix.PrefixCacheMatchInfo{},
		attrconcurrency.InFlightLoadKey:      attrconcurrency.InFlightLoad{},
	}
}

func (p *Plugin) isColdRequest(endpoints []fwksched.Endpoint) bool {
	for _, ep := range endpoints {
		attr, ok := ep.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			continue
		}
		info, ok := attr.(*attrprefix.PrefixCacheMatchInfo)
		if ok && info.MatchBlocks() > 0 {
			return false
		}
	}
	return true
}

func (p *Plugin) Filter(ctx context.Context, cycleState *fwksched.CycleState, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)

	if len(endpoints) <= 1 {
		return endpoints
	}

	isCold := p.isColdRequest(endpoints)

	// Store the cold request state in plugin state for PreRequest to update the LRU cache.
	coldState := &coldRequestState{isCold: isCold}
	p.pluginState.Write(request.RequestID, fwkplugin.StateKey(p.typedName.String()), coldState)

	// If it's a warm request, this filter does nothing (leaves it to the PrefixCacheAffinityFilter).
	if !isCold {
		return endpoints
	}

	// It's a cold request. Determine the LRU rank for all available endpoints.
	lruKeys := p.lruCache.Keys()
	lruPosition := make(map[string]int, len(lruKeys))
	for i, key := range lruKeys {
		lruPosition[key] = i // rank 0 is oldest (least recently used)
	}

	var neverUsed []fwksched.Endpoint
	var used []fwksched.Endpoint

	for _, ep := range endpoints {
		name := ep.GetMetadata().NamespacedName.String()
		if _, exists := lruPosition[name]; exists {
			used = append(used, ep)
		} else {
			neverUsed = append(neverUsed, ep)
		}
	}

	// Identify the group of "best" endpoints based on LRU.
	var bestGroup []fwksched.Endpoint
	if len(neverUsed) > 0 {
		// Any pod that has never been used for a cold request takes top priority.
		bestGroup = neverUsed
	} else {
		// Otherwise, find the exact least recently used endpoint(s).
		bestRank := math.MaxInt32
		for _, ep := range used {
			name := ep.GetMetadata().NamespacedName.String()
			rank := lruPosition[name]
			if rank < bestRank {
				bestRank = rank
				bestGroup = []fwksched.Endpoint{ep}
			} else if rank == bestRank {
				bestGroup = append(bestGroup, ep)
			}
		}
	}

	// Evaluate load gates: if our narrowed LRU pods are suffering heavy load, break the lock.
	if p.config.MaxTTFTPenaltyMs > 0 {
		bestLRUTTFT := bestTTFT(bestGroup)
		bestOverallTTFT := bestTTFT(endpoints)
		if bestLRUTTFT-bestOverallTTFT > p.config.MaxTTFTPenaltyMs {
			logger.V(logutil.DEBUG).Info("NoHitLRUFilter: TTFT load gate broken for cold request",
				"bestLRUTTFT", bestLRUTTFT, "bestOverallTTFT", bestOverallTTFT, "penalty", p.config.MaxTTFTPenaltyMs)
			return endpoints
		}
	}

	if p.config.MaxTokensInFlightPenalty > 0 {
		bestLRUTokens := bestInFlightTokens(bestGroup)
		bestOverallTokens := bestInFlightTokens(endpoints)
		if bestLRUTokens-bestOverallTokens > p.config.MaxTokensInFlightPenalty {
			logger.V(logutil.DEBUG).Info("NoHitLRUFilter: TokensInFlight load gate broken for cold request",
				"bestLRUTokens", bestLRUTokens, "bestOverallTokens", bestOverallTokens, "penalty", p.config.MaxTokensInFlightPenalty)
			return endpoints
		}
	}

	logger.V(logutil.DEBUG).Info("NoHitLRUFilter: narrowed cold request to least recently used endpoints",
		"kept", len(bestGroup), "total", len(endpoints))

	// Passed all load checks, successfully drop the recently used endpoints!
	return bestGroup
}

func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		return
	}

	coldState, err := fwkplugin.ReadPluginStateKey[*coldRequestState](p.pluginState, request.RequestID, fwkplugin.StateKey(p.typedName.String()))
	if err == nil {
		p.pluginState.Delete(request.RequestID)
	}

	if err != nil || !coldState.isCold {
		return
	}

	if targetProfile, ok := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]; ok && targetProfile != nil && len(targetProfile.TargetEndpoints) != 0 {
		p.moveTargetPodToFront(targetProfile.TargetEndpoints[0])
	}
	if targetProfile, ok := schedulingResult.ProfileResults[defaultPrefillProfile]; ok && targetProfile != nil && len(targetProfile.TargetEndpoints) != 0 {
		p.moveTargetPodToFront(targetProfile.TargetEndpoints[0])
	}
}

func (p *Plugin) moveTargetPodToFront(ep fwksched.Endpoint) {
	name := ep.GetMetadata().NamespacedName.String()
	var dummy struct{}
	p.lruCache.Add(name, dummy)
}

// helper functions ...
func bestTTFT(endpoints []fwksched.Endpoint) float64 {
	best := math.MaxFloat64
	for _, ep := range endpoints {
		if raw, ok := ep.Get(attrlatency.LatencyPredictionInfoKey); ok {
			if info, ok := raw.(*attrlatency.LatencyPredictionInfo); ok && info != nil {
				if info.TTFT() < best {
					best = info.TTFT()
				}
			}
		}
	}
	return best
}

func bestInFlightTokens(endpoints []fwksched.Endpoint) int64 {
	best := int64(math.MaxInt64)
	for _, ep := range endpoints {
		var tokens int64
		if raw, ok := ep.Get(attrconcurrency.InFlightLoadKey); ok {
			if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
				tokens = load.Tokens
			}
		}
		if tokens < best {
			best = tokens
		}
	}
	return best
}
