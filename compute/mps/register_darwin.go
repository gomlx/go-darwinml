// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mps

import "github.com/gomlx/compute"

func init() {
	compute.Register(BackendName, New)
}
