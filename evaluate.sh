NAME="./experience/overfit/checkpoints/step-400.pt"
OUT_DIR="./experience/"
SEED=1024

# Data arguments
DATA_NAME="ade20k"
DATA_SIZE=64

# Training arguments
PATCH_SIZE=16
IMG_SIZE=512
LABEL_SIZE=128
BATCH_SIZE=16
LR=1e-5
DROPOUT=0
DEVICE="cuda"
FP16=False


python evaluate.py --name $NAME --out_dir $OUT_DIR --seed $SEED \
    --data_name $DATA_NAME --data_size $DATA_SIZE \
    --patch_size $PATCH_SIZE --img_size $IMG_SIZE --label_size $LABEL_SIZE \
    --batch_size $BATCH_SIZE --dropout $DROPOUT --device $DEVICE --fp16 $FP16