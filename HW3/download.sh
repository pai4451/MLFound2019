#!/usr/bin/env bash

train=https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat
test=https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat


mkdir data
wget "${train}" -O ./data/hw3_train.dat
wget "${test}" -O ./data/hw3_test.dat