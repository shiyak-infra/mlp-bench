FROM golang:1.24.3 AS builder

WORKDIR /src
COPY . .
RUN make build

FROM nvcr.io/nvidia/pytorch:25.04-py3

COPY bench/scripts /bench/scripts
COPY --from=builder /src/bin/mlp-bench /bench/mlp-bench

RUN chmod -R +x /bench/scripts

WORKDIR /bench
