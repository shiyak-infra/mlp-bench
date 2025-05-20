ARCH        ?=amd64
TARGET_OS   ?=linux

GO_BUILD_VARS       = GO111MODULE=on GOOS=$(TARGET_OS) GOARCH=$(ARCH)
DOCKER_PLATFORM     = $(TARGET_OS)/$(ARCH)

VERSION ?= $(shell git tag --sort=committerdate | tail -1 | cut -d"v" -f2)
IMG ?= test:${VERSION}

build:
	${GO_BUILD_VARS} go build -o ./bin/mlp-bench ./

docker-build: build
	docker buildx build -t ${IMG} --platform ${DOCKER_PLATFORM} .