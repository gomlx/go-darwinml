// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package godarwinml provides GoMLX compute backends for Apple Silicon (macOS).
//
// Currently available backends:
//
//   - github.com/gomlx/go-darwinml/compute/mps ("mps"): Metal Performance Shaders Graph (MPSGraph)
//     backend for GPU computation on Apple Silicon.
//
// # Requirements
//
//   - macOS 13.0+ (Ventura or later, macOS 15.0+ recommended for scatter operations)
//   - Xcode Command Line Tools
//   - Go 1.27+
package godarwinml
