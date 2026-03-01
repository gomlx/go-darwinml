// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && !cgo

package mpsgraph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// New returns an error when CGO is disabled.
func New(config string) (backends.Backend, error) {
	return nil, errors.New("MPSGraph backend requires CGO to be enabled; rebuild with CGO_ENABLED=1")
}
