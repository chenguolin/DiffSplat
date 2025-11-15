NUM_MACHINES=1
NUM_LOCAL_GPUS=8
MACHINE_RANK=0
MAIN_MACHINE_IP=""  # fill your machine IP here
MAIN_MACHINE_PROT=""  # fill your machine port here

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TORCH_HOME=~/.cache/torch
export NCCL_DEBUG=VERSION

# Ensure local imports like `import src.*` work when invoked from repo root
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/src"

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_MACHINE_IP \
    --main_process_port $MAIN_MACHINE_PROT \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
$@
