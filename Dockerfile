FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN cd / && \
    git clone https://github.com/wilicc/gpu-burn.git && \
    cd gpu-burn && \
    make

RUN pip install lightning

COPY . /mlp-bench/

WORKDIR /mlp-bench
