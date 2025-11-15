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

python3 ${FILE} \
    --config_file ${CONFIG_FILE} \
    --tag ${TAG} \
    --allow_tf32 \
$@
