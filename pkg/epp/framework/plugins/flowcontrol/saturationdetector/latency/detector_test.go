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

package latency

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

func newTestDetector() *detector {
	return &detector{
		typedName:                    fwkplugin.TypedName{Type: LatencyDetectorType, Name: "test"},
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey,
	}
}

// endpointWith builds an endpoint carrying the given prediction. A nil info
// leaves the LatencyPredictionInfo attribute unset (never-scored endpoint).
func endpointWith(name string, info *attrlatency.LatencyPredictionInfo) datalayer.Endpoint {
	ep := datalayer.NewEndpoint(&datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}, nil)
	if info != nil {
		ep.GetAttributes().Put(attrlatency.LatencyPredictionInfoDataKey.String(), info)
	}
	return ep
}

// withinSLO: both predictions comfortably under their SLO.
func withinSLO() *attrlatency.LatencyPredictionInfo {
	return attrlatency.NewLatencyPredictionInfo(true, true, 20, 5, 80, 10, 0)
}

// violatesTTFT: TTFT SLO=100 (headroom -50 + ttft 150) is missed; TPOT met.
func violatesTTFT() *attrlatency.LatencyPredictionInfo {
	return attrlatency.NewLatencyPredictionInfo(false, true, -50, 5, 150, 10, 0)
}

// violatesTPOT: TPOT SLO=35 (headroom -5 + tpot 40) is missed; TTFT met.
func violatesTPOT() *attrlatency.LatencyPredictionInfo {
	return attrlatency.NewLatencyPredictionInfo(true, false, 20, -5, 80, 40, 0)
}

// noSLO: no SLO was set, so headroom = -prediction and both sums are zero.
// The producer marks such predictions invalid, but the detector must not count
// them as violations.
func noSLO() *attrlatency.LatencyPredictionInfo {
	return attrlatency.NewLatencyPredictionInfo(false, false, -150, -10, 150, 10, 0)
}

func TestDetector_Saturation(t *testing.T) {
	tests := []struct {
		name      string
		endpoints []datalayer.Endpoint
		want      float64
	}{
		{
			name:      "empty pool is fully saturated",
			endpoints: nil,
			want:      1.0,
		},
		{
			name:      "all within SLO",
			endpoints: []datalayer.Endpoint{endpointWith("a", withinSLO()), endpointWith("b", withinSLO())},
			want:      0.0,
		},
		{
			name:      "all violate (TTFT)",
			endpoints: []datalayer.Endpoint{endpointWith("a", violatesTTFT()), endpointWith("b", violatesTTFT())},
			want:      1.0,
		},
		{
			name:      "TPOT violation counts",
			endpoints: []datalayer.Endpoint{endpointWith("a", violatesTPOT())},
			want:      1.0,
		},
		{
			name: "half the pool violates",
			endpoints: []datalayer.Endpoint{
				endpointWith("a", violatesTTFT()),
				endpointWith("b", withinSLO()),
				endpointWith("c", violatesTPOT()),
				endpointWith("d", withinSLO()),
			},
			want: 0.5,
		},
		{
			name:      "no SLO set does not count as violation",
			endpoints: []datalayer.Endpoint{endpointWith("a", noSLO()), endpointWith("b", noSLO())},
			want:      0.0,
		},
		{
			name: "missing prediction does not count but stays in denominator",
			endpoints: []datalayer.Endpoint{
				endpointWith("a", violatesTTFT()),
				endpointWith("b", nil),
				endpointWith("c", nil),
				endpointWith("d", nil),
			},
			want: 0.25,
		},
		{
			name: "no-SLO endpoint dilutes a violation",
			endpoints: []datalayer.Endpoint{
				endpointWith("a", violatesTTFT()),
				endpointWith("b", noSLO()),
			},
			want: 0.5,
		},
	}

	d := newTestDetector()
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.InDelta(t, tc.want, d.Saturation(context.Background(), tc.endpoints), 1e-9)
		})
	}
}

func TestDetector_Saturation_SkipsNilEndpoints(t *testing.T) {
	d := newTestDetector()
	endpoints := []datalayer.Endpoint{nil, endpointWith("a", violatesTTFT()), nil}
	// Two of three slots are nil and skipped in the numerator, but the
	// denominator is the full candidate count.
	require.InDelta(t, 1.0/3.0, d.Saturation(context.Background(), endpoints), 1e-9)
}
