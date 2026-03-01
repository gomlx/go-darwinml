// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin

// Package mpsgraph implements a GoMLX backend using Apple's MPSGraph framework.
// This stub is used on non-macOS platforms.
package mpsgraph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mpsgraph"

// New returns an error on non-macOS platforms.
func New(config string) (backends.Backend, error) {
	return nil, errors.New("MPSGraph backend is only available on macOS (darwin)")
}
