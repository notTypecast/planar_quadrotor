#!/bin/bash

# using TBB
USE_TBB=true
header_path="/usr/include/tbb"
lib_path="/usr/lib/x86_64-linux-gnu"

comm="g++ -I/usr/include/eigen3/ -I. -I./simple_nn/src main_cem.cpp -I./algevo/src -o main_cem -O3"

if [ "$USE_TBB" = true ]; then
	comm+=" -ltbb -I$header_path -L$lib_path -DUSE_TBB=true -DUSE_TBB_ONEAPI=true"
fi

eval "$comm"
