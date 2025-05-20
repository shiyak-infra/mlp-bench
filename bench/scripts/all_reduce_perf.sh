#!/bin/bash

all_reduce_perf -b 4G -e 4G -i 100 -g 8 -i 100 | grep "Avg bus bandwidth" | sed 's/^# *//'