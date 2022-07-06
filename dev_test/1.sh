#!/bin/bash
rm -rf /tmp/cpp_fifo
mkdir /tmp/cpp_fifo

for i in {0..100}
do
  mkfifo /tmp/cpp_fifo/"$i"_0
  mkfifo /tmp/cpp_fifo/"$i"_1
done
