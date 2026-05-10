module github.com/gomlx/go-darwinml

go 1.25.5

require (
	github.com/gomlx/go-darwinml/mpsgraph/gomlx v0.0.0
	github.com/google/uuid v1.6.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	github.com/x448/float16 v0.8.4
	google.golang.org/protobuf v1.36.11
)

replace github.com/gomlx/go-darwinml/mpsgraph/gomlx => ./mpsgraph/gomlx

require (
	github.com/go-logr/logr v1.4.3 // indirect
	golang.org/x/exp v0.0.0-20260312153236-7ab1446f8b90 // indirect
	k8s.io/klog/v2 v2.140.0 // indirect
)

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/gomlx/gomlx v0.27.3
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
