# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=63667 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-29500}
# WORK_DIR="/home/sahithi_kukkala/gr2html/Rodla/Rodla_result"
# CHECKPOINT_PATH="/home/sahithi_kukkala/gr2html/Rodla/rodla_internimage_xl_doclaynet.pth"  # Replace this with the path to your checkpoint file


# CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=63667 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#     --resume-from /ssd_scratch/sahithi/rodla_internimage_xl_doclaynet.pth --work-dir /home/sahithi_kukkala/gr2html/Rodla/Rodla_result\
    
#!/usr/bin/env bash

CONFIG=$1
WORK_DIR="/home/sahithi_kukkala/gr2html/Rodla/Rodla_result"
CHECKPOINT_PATH="/home/sahithi_kukkala/gr2html/Rodla/rodla_internimage_xl_doclaynet.pth"

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG \
    --resume-from $CHECKPOINT_PATH \
    --work-dir $WORK_DIR
