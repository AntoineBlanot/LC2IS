from tqdm import tqdm
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

class Engine():
    def __init__(self, name: str, 
        model: nn.Module, optimizer: nn.optim = None, criterion: nn = None, lr_scheduler: optim.lr_scheduler = None,
        device = "cuda", fp16: bool = True,
        train_loader : DataLoader = None, eval_loader: DataLoader = None, compute_metrics: Callable = None,
        max_epoch: int = 1, max_steps: int = 500, eval_step: Union[str, int] = "epoch", log_step: Union[str, int] = "epoch",
        save_step: Union[str, int] = "epoch", save_dir: str = "./checkpoints/",
        logger: str = "wandb", logger_args: Dict = None
    ) -> None:

        self.name = name

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.fp16 = fp16

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.compute_metrics = compute_metrics

        self.steps_in_epoch = len(train_loader)
        self.train_steps = max(self.steps_in_epoch*max_epoch, max_steps)
        self.eval_step = self.steps_in_epoch if eval_step == "epoch" else eval_step
        self.log_step = self.steps_in_epoch if log_step == "epoch" else log_step
        self.save_step = self.steps_in_epoch if save_step == "epoch" else save_step
        self.save_dir = save_dir
        
        self.logger = logger
        if logger == "wandb":
            import wandb
            wandb.init(**logger_args)
            wandb.watch(self.model, log_freq=log_step)
    
    def train(self) -> Tuple[Dict[str, float], str]:
        self.train_progress = tqdm(range(self.train_steps), desc="Training")

        self.stop_train = False
        self.train_metrics, self.all_train_loss, self.train_step = {}, [], 0

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.model.train()
        while not self.stop_train:
            metrics, save_path = self.train_loop()

        return metrics, save_path

    def train_loop(self):
        for data in self.train_loader:
            self.train_step += 1

            inputs, mappings, originals = data
            inputs = {k: v.to(self.device) for k,v in inputs.items()}

            self.optimizer.zero_grad()
            if self.fp16:
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    _, _, outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs["label"])

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, _, outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs["label"])
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.train_step += 1
            self.train_progress.update()

            self.all_train_loss.append(loss.item())

            eval_metrics = self.should_eval()
            log_metrics = self.should_log(eval_metrics=eval_metrics)
            save_path = self.should_save()
            stop = self.should_stop()

            if stop:
                self.stop_train = True
                break

        return log_metrics, save_path

    def evaluate(self) -> Tuple[float, Any]:
        self.model.eval()

        eval_steps = len(self.eval_loader)
        eval_progress = tqdm(range(eval_steps), desc="Evaluation", leave=False)

        all_loss = []
        all_out, all_gt, all_size = None, None, None

        for data in self.eval_loader:

            inputs, mappings, originals = data
            inputs = {k: v.to(self.device) for k,v in inputs.items()}

            with torch.no_grad():
                _, _, outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs["label"])

            eval_progress.update()

            all_loss.append(loss.item())
            all_out = torch.concat([all_out, outputs]) if all_out is not None else outputs
            all_gt = all_gt + originals["label"] if all_gt is not None else originals["label"]
            all_size = torch.concat([all_size, inputs["size"]]) if all_size is not None else inputs["size"]

        eval_loss = np.array(all_loss).mean()
        return (eval_loss), all_out, all_gt, all_size

    def log(self, eval_metrics: Dict[str, float]) -> Dict[str, float]:
        train_step = self.train_step
        train_epoch = round(self.train_step / self.steps_in_epoch, 4)
        train_loss = np.array(self.all_train_loss).mean()
        metrics = {
            **dict(train_step=train_step, train_epoch=train_epoch, train_loss=train_loss),
            **eval_metrics
        }

        self.train_progress.set_postfix(metrics)
        if self.logger == "wandb":
            wandb.log({"/".join(k.split("_")): v for k,v in metrics.items()})

        return metrics

    def save(self) -> str:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir + self.name + "/step-" + self.train_step + ".pt"
        torch.save(self.model.state_dict(), save_path)
        
        return save_path


    def should_eval(self) -> Dict[str, float]:
        if self.eval_loader is not None and (self.train_step == self.eval_step):
            eval_loss, eval_outputs = self.evaluate()
            self.model.train()
            if self.compute_metrics is not None:
                eval_metrics = self.compute_metrics(eval_outputs)
                return {**dict(eval_loss=eval_loss), **eval_metrics}
            else:
                return dict(eval_loss=eval_loss)
        else:
            return dict()

    def should_log(self, eval_metrics: Dict[str, float]) -> Dict[str, float]:
        if self.train_step == self.log_step:
            metrics = self.log(eval_metrics=eval_metrics)
            self.all_train_loss = []
            return metrics
        else:
            return None
    
    def should_save(self) -> str:
        if self.train_step == self.save_step:
            save_path = self.save()
            return save_path
        else:
            return None

    def should_stop(self) -> bool:
        if self.train_step == self.train_steps:
            return True
        else:
            False
