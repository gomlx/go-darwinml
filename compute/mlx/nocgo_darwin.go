// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && !cgo

package mlx

import (
	"github.com/gomlx/compute"
	"github.com/pkg/errors"
)

// BackendName is the name used to register this backend.
const BackendName = "mlx"

// New returns an error when CGO is disabled.
func New(config string) (compute.Backend, error) {
	return nil, errors.New("MLX backend requires CGO to be enabled; rebuild with CGO_ENABLED=1")
}
