#!/bin/sh
#PBS -q rt_HF
#PBS -N Qwen3-30B-A3B
#PBS -l select=16:ncpus=192:mpiprocs=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m n
#PBS -koed
#PBS -V
#PBS -o outputs/Qwen-3-30B-A3B/

cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.9.1
module load cudnn/9.10.2
module load nccl/2.27.5-cuda12.9
module load hpcx/2.20

source .venv/bin/activate

# distributed settings
JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

# model config
# https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
HIDDEN_SIZE=2048
MOE_FFN_HIDDEN_SIZE=768
NUM_LAYERS=48
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=4
SEQ_LENGTH=32768
MAX_POSITION_EMBEDDINGS=32768

HEAD_DIMENSION=128

# distributed settings
TENSOR_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1

CONTEXT_PARALLEL_SIZE=2

DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

PIPELINE_MODEL_CHUNKS=1
LAYERS_PER_VIRTUAL_PIPELINE_STAGE=$((${NUM_LAYERS} / ${PIPELINE_PARALLEL_SIZE} / ${PIPELINE_MODEL_CHUNKS}))

TP_COMM_OVERLAP=""
if [[ $TENSOR_PARALLEL_SIZE -gt 1 ]]; then
  TP_COMM_OVERLAP="
    --tp-comm-overlap \
    --tp-comm-bootstrap-backend nccl
  "
fi

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
TRAIN_STEPS=25000  # 200B tokens
LR_DECAY_ITERS=25000

LR=2.50E-5
MIN_LR=2.50E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Qwen3-8B-Base/
CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Megatron-Bridge/Qwen3-30B-A3B-Base
CHECKPOINT_SAVE_DIR=/groups/gch51639/fujii/checkpoints/Qwen-3-30B-A3B-Base/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-ep${EXPERT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}

echo "Checkpoint save dir: ${CHECKPOINT_SAVE_DIR}"
mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 2141101288 /groups/gag51395/datasets/Qwen-2-Tokenizer/Japanese/ja_wikipedia_2503_text_document"

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
  echo "No checkpoint found in ${CHECKPOINT_SAVE_DIR}, starting from ${CHECKPOINT_DIR}"
fi

# interleaved pipeline
PIPELINE_ARGS="--pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}"

if [[ ${PIPELINE_MODEL_CHUNKS} -gt 1 ]]; then
  echo "Interleaved pipeline is enabled: layers per virtual pipeline stage = ${LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"

  PIPELINE_ARGS="${PIPELINE_ARGS} --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"
fi

# limit cuda synch acceleration
ACCELERATION_OPTIONS=True
ACCELERATION_ARGS=""
if [[ ${ACCELERATION_OPTIONS} == "True" ]]; then
  echo "Acceleration options are enabled"

  ACCELERATION_ARGS="
  --no-log-loss-scale-to-tensorboard \
  --no-create-attention-mask-in-dataloader \
  "
fi

# FP8 Acceleration
BLOCKWISE_FP8=False
FP8_PARAM_GATHER=False
USE_PRECISION_AWARE_OPTIMIZER=False
if [[ ${BLOCKWISE_FP8} == "True" ]]; then
  echo "Blockwise FP8 is enabled"
  export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
  ACCELERATION_ARGS="${ACCELERATION_ARGS} --fp8-format e4m3 --fp8-recipe blockwise"

  if [[ ${FP8_PARAM_GATHER} == "True" ]]; then
    echo "FP8 parameter gather is enabled"
    ACCELERATION_ARGS="${ACCELERATION_ARGS} --fp8-param-gather "
  fi

  if [[ ${USE_PRECISION_AWARE_OPTIMIZER} == "True" ]]; then
    echo "Precision aware optimizer is enabled"
      ACCELERATION_ARGS="${ACCELERATION_ARGS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
  fi
fi

mpirun --display-allocation --display-map --report-bindings \
  -np $NUM_GPUS \
  --map-by ppr:$NUM_GPU_PER_NODE:node \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -bind-to none \
  python pretrain_gpt.py \
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE} \
    ${PIPELINE_ARGS} \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    ${TP_COMM_OVERLAP} \
    --distributed-timeout-minutes 30 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size 6144 \
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --kv-channels ${HEAD_DIMENSION} \
    --group-query-attention \
    --num-query-groups ${NUM_KEY_VALUE_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_STEPS} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${CHECKPOINT_ARGS} \
    --save ${CHECKPOINT_SAVE_DIR} \
    --data-path ${TRAIN_DATA_PATH} \
    --split 990,10,0 \
    --distributed-backend nccl \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad ${GRAD_CLIP} \
    --lr-warmup-iters ${LR_WARMUP_STEPS} \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --log-interval 1 \
    --save-interval 500 \
    --ckpt-format "torch_dist" \
    --async-save \
    --eval-interval 500 \
    --eval-iters 10 \
    --bf16 \
    --no-initialization \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-base 1000000.0 \
    --rotary-percent 1.0 \
    --make-vocab-size-divisible-by 1187 \
    --disable-bias-linear \
    --use-mcore-models \
    --no-rope-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --no-masked-softmax-fusion \
    --no-bias-swiglu-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --qk-layernorm \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-dtype fp32 \
    --moe-aux-loss-coeff 1e-3 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --moe-grouped-gemm \
    --moe-layer-recompute \
    --moe-permute-fusion \
    --use-flash-attn \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --transformer-impl "transformer_engine" \
    --use-mpi \
    --log-memory-to-tensorboard \
    ${ACCELERATION_ARGS} \
    --log-throughput
