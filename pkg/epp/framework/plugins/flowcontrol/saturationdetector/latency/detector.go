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

// Package latency implements a saturation detector driven by predicted-latency
// SLO attainment. It consumes the LatencyPredictionInfo attribute produced by
// the predicted-latency plugin and reports pool saturation as the fraction of
// candidate endpoints whose predicted TTFT or TPOT is expected to miss its SLO.
//
// Unlike sloheadroomtier (a scheduling filter that picks endpoints for a single
// request), this detector feeds flow-control admission and the pool saturation
// gauge: it answers "is the pool as a whole predicted to miss SLO" rather than
// "where should this request go".
package latency

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

const (
	// LatencyDetectorType is the unique identifier for this plugin.
	LatencyDetectorType = "latency-detector"
)

// Config is the external configuration schema for the latency detector.
type Config struct {
	// LatencyPredictionInfoProducerName selects which predicted-latency
	// producer's LatencyPredictionInfo to read. Empty uses the default.
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
}

// LatencyDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func LatencyDetectorFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var cfg Config
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal latency detector config: %w", err)
		}
	}

	typedName := fwkplugin.TypedName{Type: LatencyDetectorType, Name: name}
	log.FromContext(handle.Context()).WithName(typedName.String()).V(logutil.DEFAULT).Info("Creating new LatencyDetector",
		"latencyPredictionInfoProducerName", cfg.LatencyPredictionInfoProducerName)

	return &detector{
		typedName:                    typedName,
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(cfg.LatencyPredictionInfoProducerName),
	}, nil
}

var _ flowcontrol.SaturationDetector = &detector{}

// detector reports pool saturation from predicted-latency SLO attainment.
type detector struct {
	typedName                    fwkplugin.TypedName
	latencyPredictionInfoDataKey fwkplugin.DataKey
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

func (d *detector) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{d.latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfo{}},
	}
}

// Saturation returns the fraction of candidate endpoints predicted to miss SLO.
//
//	Saturation = (# endpoints whose predicted TTFT or TPOT exceeds its SLO) / (# candidates)
//
// An endpoint with no prediction, or whose request carried no SLO, contributes
// 0 (no latency backpressure). The signal therefore reaches 1.0 only when every
// candidate is predicted to miss SLO. An empty candidate set is treated as fully
// saturated, matching the other saturation detectors.
func (d *detector) Saturation(_ context.Context, endpoints []datalayer.Endpoint) float64 {
	if len(endpoints) == 0 {
		return 1.0
	}

	var violating int
	for _, e := range endpoints {
		if e == nil || e.GetMetadata() == nil {
			continue
		}
		if info := d.getPrediction(e.GetAttributes()); info != nil && violatesSLO(info) {
			violating++
		}
	}

	return float64(violating) / float64(len(endpoints))
}

func (d *detector) getPrediction(m datalayer.AttributeMap) *attrlatency.LatencyPredictionInfo {
	if m == nil {
		return nil
	}
	if v, ok := m.Get(d.latencyPredictionInfoDataKey.String()); ok {
		if info, ok := v.(*attrlatency.LatencyPredictionInfo); ok {
			return info
		}
	}
	return nil
}

// violatesSLO reports whether a prediction misses a configured TTFT or TPOT SLO.
//
// The predicted-latency producer sets headroom = SLO - prediction, so the SLO is
// recoverable as headroom + prediction and is exactly zero when no SLO was set
// (x + (-x) == 0 in IEEE arithmetic). A dimension is only counted when its SLO
// is set, so SLO-less traffic never registers as a violation.
func violatesSLO(info *attrlatency.LatencyPredictionInfo) bool {
	ttftSLOSet := info.TTFTHeadroom()+info.TTFT() > 0
	tpotSLOSet := info.TPOTHeadroom()+info.TPOT() > 0
	return (ttftSLOSet && !info.TTFTValid()) || (tpotSLOSet && !info.TPOTValid())
}
