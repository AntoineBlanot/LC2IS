import argparse
from importlib import import_module

import torch
from torch.utils.data import DataLoader

from engine import Engine
from metrics import meanIOU


def get_config() -> tuple[dict, dict, dict, dict, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Run name")
    args = parser.parse_args()

    config_module = import_module(name=args.config)
    train_config = config_module.train_config
    data_config = config_module.data_config
    model_config = config_module.model_config
    optim_config = config_module.optim_config
    lr_config = config_module.lr_config

    return train_config, data_config, model_config, optim_config, lr_config

train_config, data_config, model_config, optim_config, lr_config = get_config()
torch.manual_seed(train_config.pop("seed"))

#region dataset and data loaders
collator = data_config.get("collator").pop("type")(
    **data_config.pop("collator")
)
train_data = data_config.get("training").pop("type")(
    split="training",
    **data_config.pop("training")
)
eval_data = data_config.get("validation").pop("type")(
    split="validation",
    **data_config.pop("validation")
)
train_loader = DataLoader(dataset=train_data, batch_size=train_config.pop("train_batch_size"), collate_fn=collator)
eval_loader = DataLoader(dataset=eval_data, batch_size=train_config.pop("eval_batch_size"), collate_fn=collator, shuffle=False)
#endregion

#region model and criterion
model = model_config.pop("type")(
    class_names=train_data.class_names,
    **model_config
)
criterion = torch.nn.CrossEntropyLoss()
#endregion

#region optimizer and schedulers
optim = optim_config.pop("type")(
    params=[
        {**dict(params=getattr(model, p.pop("key")).parameters()), **p}
        for p in optim_config.pop("params")
    ],
    **optim_config
)
lr_scheduler = lr_config.pop("type")(
    optimizer=optim,
    schedulers=[
        p.pop("type")(optimizer=optim, **p)
        for p in lr_config.pop("schedulers")
    ],
    **lr_config
)
#endregion

#region logger
logger_name = train_config.get("logger").pop("type")
logger_args = train_config.pop("logger")
#endregion


# Define engine
trainer = Engine(
    **train_config,
    model=model, optimizer=optim, criterion=criterion, lr_scheduler=lr_scheduler,
    train_loader=train_loader, eval_loader=eval_loader, compute_metrics=meanIOU,
    logger=logger_name, logger_args=logger_args
)

# Training
train_metrics, save_dir = trainer.train()

print(train_metrics)
print("Model checkpoints saved at path: {}".format(save_dir))
