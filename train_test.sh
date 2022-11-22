# Base arguments
NAME="contrastive-test"
OUT_DIR="./experience/"
SEED=1024

# Data arguments
DATA_SIZE=640

# Training arguments
PATCH_SIZE=16
IMG_SIZE=224
LABEL_SIZE=56
BATCH_SIZE=8
LR=1e-5
DROPOUT=0
WEIGHT_DECAY=0
DEVICE="cuda"
MAX_EPOCH=50

# Logging arguments
WANDB_PROJECT="LC2IS-exp"

python train_test.py --name $NAME --out_dir $OUT_DIR --seed $SEED \
    --data_size $DATA_SIZE \
    --patch_size $PATCH_SIZE --img_size $IMG_SIZE --label_size $LABEL_SIZE \
    --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --device $DEVICE \
    --max_epoch $MAX_EPOCH \
    --wandb_project $WANDB_PROJECT
