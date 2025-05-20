ARCH        ?=amd64
TARGET_OS   ?=linux

GO_BUILD_VARS       = GO111MODULE=on GOOS=$(TARGET_OS) GOARCH=$(ARCH)
DOCKER_PLATFORM     = $(TARGET_OS)/$(ARCH)

VERSION ?= $(shell git tag --sort=committerdate | tail -1 | cut -d"v" -f2)
IMG ?= ghcr.io/shiyak-infra/mlp-bench:${VERSION}

build:
	${GO_BUILD_VARS} go build -o ./bin/mlp-bench ./

docker-build:
	docker build -t ${IMG} --platform ${DOCKER_PLATFORM} .

docker-push:
	docker push ${IMG}