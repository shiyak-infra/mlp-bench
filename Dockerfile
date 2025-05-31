FROM golang:1.24.3 AS builder

WORKDIR /src
COPY . .
RUN make build

FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN cd / && \
    git clone https://github.com/wilicc/gpu-burn.git && \
    cd gpu-burn && \
    make

RUN pip install lightning

COPY benchs /mlp-bench/benchs
COPY main.py /mlp-bench/main.py
COPY config.json /mlp-bench/config.json

WORKDIR /mlp-bench

ENTRYPOINT ["python", "main.py"]
