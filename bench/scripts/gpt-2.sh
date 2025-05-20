#!/bin/bash

mpirun --tag-output --allow-run-as-root -np 8 --hostfile /job/hostfile -map-by slot -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=WARN -x NCCL_IB_QPS_PER_CONNECTION=4 python ./gpt-2/ddp.py