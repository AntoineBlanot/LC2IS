# Base arguments
NAME="base-512"
OUT_DIR="./experience/"
SEED=1024

# Data arguments
JOIN=" and "

# Training arguments
PATCH_SIZE=16
IMG_SIZE=512
LABEL_SIZE=128
BATCH_SIZE=16
LR=1e-5
DROPOUT=0
WEIGHT_DECAY=0
DEVICE="cuda"
FP16=False
MAX_EPOCH=50

# Logging arguments
WANDB_PROJECT="LC2IS-exp"

python train.py --name $NAME --out_dir $OUT_DIR --seed $SEED \
    --join $JOIN \
    --patch_size $PATCH_SIZE --img_size $IMG_SIZE --label_size $LABEL_SIZE \
    --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --device $DEVICE --fp16 $FP16 \
    --max_epoch $MAX_EPOCH \
    --wandb_project $WANDB_PROJECT
