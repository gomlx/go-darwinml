// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build darwin && cgo

package mpsgraph

import "github.com/gomlx/gomlx/backends"

func init() {
	backends.Register(BackendName, New)
}
