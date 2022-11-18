# Base arguments
NAME="default-name"
OUT_DIR="./experience/"
SEED=1024

# Data arguments
DATA_NAME="ade20k"
DATA_SIZE=100

# Training arguments
PATCH_SIZE=16
IMG_SIZE=512
LABEL_SIZE=128
BATCH_SIZE=16
LR=1e-5
DROPOUT=0
WEIGHT_DECAY=0
DEVICE="cpu"
FP16=False
MAX_EPOCH=5
MAX_STEPS=100
LOG_STEP=100
EVAL_STEP=100
SAVE_STEP=100

# Logging arguments
WANDB_PROJECT="wandb-project"

python default.py --name $NAME --out_dir $OUT_DIR --seed $SEED \
    --data_name $DATA_NAME --data_size $DATA_SIZE \
    --patch_size $PATCH_SIZE --img_size $IMG_SIZE --label_size $LABEL_SIZE \
    --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --device $DEVICE --fp16 $FP16 \
    --max_epoch $MAX_EPOCH --max_steps $MAX_STEPS \
    --log_step $LOG_STEP --eval_step $EVAL_STEP --save_step $SAVE_STEP \
    --wandb_project $WANDB_PROJECT
