// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build !darwin

// Package mps implements a GoMLX compute backend using Apple's MPSGraph framework.
// This stub is used on non-macOS platforms.
package mps

import (
	"github.com/gomlx/compute"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mps"

// New returns an error on non-macOS platforms.
func New(config string) (compute.Backend, error) {
	return nil, errors.New("MPS backend is only available on macOS (darwin)")
}
