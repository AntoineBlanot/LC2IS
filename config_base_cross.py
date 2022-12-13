from torch import optim
import torchvision.transforms as T

from data.dataset import ADE20K
from data.collator import ADE20KCollator
from model.scheduler import PolynomialLR
from model.model import BaseCrossA

NAME = "base_cross"

train_config = dict(
    out_dir="./experience/long/",
    name=NAME,
    seed=1024,
    train_batch_size=8,
    eval_batch_size=8,
    max_steps=int(160e3),
    eval_step=int(10e3),
    log_step=int(10e3),
    save_step=int(10e3),
    device="cuda",
    logger=dict(
        type="wandb",
        project="long",
        name=NAME
    )
)

data_config = dict(
    training=dict(
        type=ADE20K,
        transform=T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=0, scale=(0.5, 2)),
            T.RandomCrop(size=512, pad_if_needed=True)
        ]),
        size=None
    ),
    validation=dict(
        type=ADE20K,
        transform=T.CenterCrop(size=512),
        size=None
    ),
    collator=dict(
        type=ADE20KCollator
    )
)

model_config = dict(
    type=BaseCrossA,
    dec_dim=512,
    dec_depth=[2, 4, 2],
    nhead=8,
    dropout=0.0
)

optim_config = dict(
    type=optim.AdamW,
    lr=1e-4,
    weight_decay=1e-4,
    params=[
        dict(key="vision_encoder", lr=1e-5),
        dict(key="vision_decoder"),
        dict(key="classes"),
    ]
)

lr_config = dict(
    type=optim.lr_scheduler.SequentialLR,
    schedulers=[
        {"type": optim.lr_scheduler.LinearLR, "start_factor": 1/1500, "total_iters": 1500},
        {"type": PolynomialLR, "power": 0.9, "total_iters": 160000-1500}
    ],
    milestones=[1500]
)
