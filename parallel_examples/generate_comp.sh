#!/bin/bash
gpus=(1 2 4)

SCRIPT="parallel_examples/run_wan_lora.py"

for val in "${gpus[@]}"; do
  echo "Running with --nproc_per_gpu=$val"
  torchrun --nproc_per_node=$val $SCRIPT 
done
