#!/bin/bash

git clone https://github.com/shiyak-infra/gpu-burn.git  > /dev/null 2>&1 && \
 cd gpu-burn > /dev/null 2>&1 && \
 make > /dev/null 2>&1 && \
 mv ./gpu_burn /usr/local/bin > /dev/null 2>&1

gpu_burn -m 100% -d -stts 10 60 | tail -n 8