#!/bin/bash

git clone https://github.com/wilicc/gpu-burn.git && cd gpu-burn && make

./gpu_burn -m 100% -d -stts 10 60 | grep 100.0% > out