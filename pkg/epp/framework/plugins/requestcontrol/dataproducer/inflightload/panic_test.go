/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestInFlightLoadProducer_PanicSafety(t *testing.T) {
	producer := newTestProducer(t)
	ctx := context.Background()

	t.Run("ExtractEndpoint", func(t *testing.T) {
		// 1. Nil Endpoint
		require.NotPanics(t, func() {
			_ = producer.ExtractEndpoint(ctx, datalayer.EndpointEvent{Type: datalayer.EventDelete, Endpoint: nil})
		})

		// 2. Nil Metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		require.NotPanics(t, func() {
			_ = producer.ExtractEndpoint(ctx, datalayer.EndpointEvent{Type: datalayer.EventDelete, Endpoint: stub})
		})
	})

	t.Run("Produce", func(t *testing.T) {
		// 1. Nil Endpoints slice
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, nil)
		})

		// 2. Slice with nil endpoint
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, []fwksched.Endpoint{nil})
		})

		// 3. Endpoint with nil metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		require.NotPanics(t, func() {
			_ = producer.Produce(ctx, nil, []fwksched.Endpoint{stub})
		})
	})

	t.Run("PreRequest", func(t *testing.T) {
		// 1. Nil Result
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, nil, nil)
		})

		// 2. Nil Request, non-nil Result
		res := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{newStubSchedulingEndpoint("ep1")}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, nil, res)
		})

		// 3. Empty ProfileResults
		resEmpty := &fwksched.SchedulingResult{ProfileResults: map[string]*fwksched.ProfileRunResult{}}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resEmpty)
		})

		// 4. Nil ProfileResult
		resNilProfile := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{"default": nil},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resNilProfile)
		})

		// 5. Empty TargetEndpoints
		resEmptyEndpoints := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resEmptyEndpoints)
		})

		// 6. Nil Endpoint in TargetEndpoints
		resNilEndpoint := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{nil}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resNilEndpoint)
		})

		// 7. Endpoint with nil metadata
		stub := newStubSchedulingEndpoint("nil-meta")
		stub.metadata = nil
		resNilMeta := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{stub}},
			},
		}
		require.NotPanics(t, func() {
			producer.PreRequest(ctx, &fwksched.InferenceRequest{}, resNilMeta)
		})
	})

	t.Run("ResponseBody", func(t *testing.T) {
		// 1. Nil Request or Response
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, nil, nil, nil)
		})
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, &fwksched.InferenceRequest{}, nil, nil)
		})
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, nil, &requestcontrol.Response{}, nil)
		})

		// 2. Nil SchedulingResult
		reqNoRes := &fwksched.InferenceRequest{SchedulingResult: nil}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNoRes, &requestcontrol.Response{}, nil)
		})

		// 3. Various nil components in result
		resNilProfile := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{"default": nil},
		}
		reqNilProfile := &fwksched.InferenceRequest{SchedulingResult: resNilProfile}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNilProfile, &requestcontrol.Response{EndOfStream: true}, nil)
		})

		resNilEndpoint := &fwksched.SchedulingResult{
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"default": {TargetEndpoints: []fwksched.Endpoint{nil}},
			},
		}
		reqNilEndpoint := &fwksched.InferenceRequest{SchedulingResult: resNilEndpoint}
		require.NotPanics(t, func() {
			producer.ResponseBody(ctx, reqNilEndpoint, &requestcontrol.Response{EndOfStream: true}, nil)
		})
	})

	t.Run("Factory_NilHandle", func(t *testing.T) {
		require.NotPanics(t, func() {
			_, _ = InFlightLoadProducerFactory("test", nil, nil)
		})
	})
}
