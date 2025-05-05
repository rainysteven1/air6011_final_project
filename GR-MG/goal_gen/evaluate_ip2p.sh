#!/usr/bin/env bash
set -x
GPUS_PER_NODE=2 # number of gpus per machine
MASTER_ADDR="localhost:12345" # modify it with your own address and port
NNODES=1 # number of machines
JOB_ID=108
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    goal_gen/evaluate.py \
    --config ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $NNODES
