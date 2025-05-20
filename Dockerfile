FROM nvcr.io/nvidia/pytorch:25.04-py3

COPY bench /bench

COPY ./bin/mlp-bench /bench/mlp-bench

WORKDIR /bench

RUN pip install -r requirements.txt

ENTRYPOINT ["mlp-bench"]