#!/usr/bin/env bash

train6=https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_6_train.dat
train7=https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_train.dat
test7=https://www.csie.ntu.edu.tw/~htlin/course/mlfound19fall/hw1/hw1_7_test.dat

mkdir data
wget "${train6}" -O ./data/hw1_6_train.dat
wget "${train7}" -O ./data/hw1_7_train.dat
wget "${test7}" -O ./data/hw1_7_test.dat