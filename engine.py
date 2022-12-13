from typing import Callable
from tqdm import tqdm
from pathlib import Path

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb


class Engine():
    def __init__(self, name: str, 
        model: nn.Module, optimizer: optim = None, criterion: nn = None, lr_scheduler: optim.lr_scheduler = None,
        device = "cuda", fp16: bool = False, aux_criterion: nn = None,
        train_loader : DataLoader = None, eval_loader: DataLoader = None, compute_metrics: Callable = None,
        max_epoch: int = 1, max_steps: int = None, eval_step: int = None, log_step: int = None, save_step: int = None,
        out_dir: str = "./", logger: str = "wandb", logger_args: dict = None
    ) -> None:

        self.name = name

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.fp16 = fp16
        self.model.to(self.device)
        self.aux_criterion = aux_criterion
        
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.compute_metrics = compute_metrics

        self.steps_in_epoch = len(train_loader) if train_loader is not None else 0
        self.train_steps = max(self.steps_in_epoch*max_epoch, max_steps) if max_steps is not None else self.steps_in_epoch*max_epoch
        self.eval_step = self.steps_in_epoch * 10 if eval_step is None else eval_step
        self.log_step = self.steps_in_epoch if log_step is None else log_step
        self.save_step = self.steps_in_epoch * 10 if save_step is None else save_step
        self.out_dir = out_dir + name + "/"
        
        self.logger = logger
        self.logger_args = logger_args
    
    def train(self) -> tuple[dict, str]:

        if self.logger == "wandb":
            wandb.init(**self.logger_args)
            wandb.watch(self.model, log_freq=self.log_step)
        self.train_progress = tqdm(range(self.train_steps), desc="Training")

        self.stop_train, self.train_step = False, 0
        self.train_metrics, self.eval_metrics = {}, {}
        self.all_train_metrics = {}

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        while not self.stop_train:
            metrics, save_path = self.train_loop()

        wandb.finish()

        return metrics, save_path

    def train_loop(self) -> tuple[dict, str]:
        self.model.train()
        for data in self.train_loader:
            self.train_step += 1

            inputs, metas = data
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            labels = inputs.pop("label")

            self.optimizer.zero_grad()
            if self.fp16:
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs_dict = self.model(inputs)
                    train_loss = self.criterion(outputs_dict["outputs"], labels)
                    losses_dict = dict(train_loss=train_loss.item())
                    if "low_score_map" in outputs_dict.keys():
                        aux_loss = self.aux_criterion(outputs_dict["low_score_map"], labels)
                        losses_dict.update(dict(aux_loss=aux_loss.item() * 0.4))
                    loss = sum([v for v in losses_dict.values()])

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs_dict = self.model(inputs)
                train_loss = self.criterion(outputs_dict["outputs"], labels)
                losses_dict = dict(train_loss=train_loss)
                if "low_score_map" in outputs_dict.keys():
                    train_aux_loss = self.aux_criterion(outputs_dict["low_score_map"], labels)
                    losses_dict.update(dict(train_aux_loss=train_aux_loss * 0.4))
                loss = torch.stack([v for v in losses_dict.values()]).sum()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.train_progress.update()
            self.all_train_metrics = {
                k: self.all_train_metrics[k] + [v.item()] if k in self.all_train_metrics.keys() else [v.item()]
                for k,v in losses_dict.items()
            }

            eval_metrics = self.should_eval()
            log_metrics = self.should_log()
            save_path = self.should_save()
            stop = self.should_stop()

            if stop:
                self.stop_train = True
                break
        
        metrics = {**log_metrics, **eval_metrics}
        
        return metrics, save_path

    def evaluate(self) -> dict:
        eval_metrics, eval_outputs = self.eval_loop()

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(**eval_outputs)
            eval_metrics = {**eval_metrics, **{"eval_"+k: v for k,v in metrics.items()}}
        
        return eval_metrics

    def eval_loop(self) -> tuple[dict, dict]:
        self.model.eval()

        eval_steps = len(self.eval_loader)
        eval_progress = tqdm(range(eval_steps), desc="Evaluation", leave=False)

        self.all_eval_metrics = {}
        all_outputs, all_labels = None, None

        for data in self.eval_loader:
            inputs, metas = data
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            labels = inputs.pop("label")

            with torch.no_grad():
                outputs_dict = self.model(inputs)
                eval_loss = self.criterion(outputs_dict["outputs"], labels)
                losses_dict = dict(eval_loss=eval_loss)
                if "low_score_map" in outputs_dict.keys():
                    eval_aux_loss = self.aux_criterion(outputs_dict["low_score_map"], labels)
                    losses_dict.update(dict(eval_aux_loss=eval_aux_loss * 0.4))

            eval_progress.update()

            self.all_eval_metrics = {
                k: self.all_eval_metrics[k] + [v.item()] if k in self.all_eval_metrics.keys() else [v.item()]
                for k,v in losses_dict.items()
            }
            all_outputs = torch.concat([all_outputs, outputs_dict["outputs"].cpu()]) if all_outputs is not None else outputs_dict["outputs"].cpu()
            all_labels = torch.concat([all_labels, labels.cpu()]) if all_labels is not None else labels.cpu()

        eval_metrics = {k: np.array(v).mean() for k,v in self.all_eval_metrics.items()}
        eval_outputs = dict(outputs=all_outputs, labels=all_labels)

        return eval_metrics, eval_outputs

    def log(self) -> dict:
        train_step = self.train_step
        train_epoch = round(self.train_step / self.steps_in_epoch, 4)
        train_metrics = {k: np.array(v).mean() for k,v in self.all_train_metrics.items()}
        metrics = {
            **dict(train_step=train_step, train_epoch=train_epoch),
            **train_metrics,
            **self.eval_metrics
        }

        self.train_progress.set_postfix(metrics)
        if self.logger == "wandb":
            wandb.log({"/".join(k.split("_")): v for k,v in metrics.items()})

        return metrics

    def save(self) -> str:
        checkpoints_dir = self.out_dir + "checkpoints/"
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoints_dir+"step-" + str(self.train_step) + ".pt")
        return checkpoints_dir

    def should_eval(self) -> dict:
        if self.eval_loader is not None and ((self.train_step % self.eval_step) == 0):
            self.eval_metrics = self.evaluate()
            self.model.train()
            return self.eval_metrics
        else:
            return {}

    def should_log(self) -> dict:
        if (self.train_step % self.log_step) == 0:
            metrics = self.log()
            self.all_train_metrics = {}
            return metrics
        else:
            return {}
    
    def should_save(self) -> str:
        if (self.train_step % self.save_step) == 0:
            save_path = self.save()
            return save_path
        else:
            return None

    def should_stop(self) -> bool:
        if (self.train_step % self.train_steps) == 0:
            return True
        else:
            False
