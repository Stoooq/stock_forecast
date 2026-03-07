import torch
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from training.checkpointer import Checkpointer

class Trainer:
    def __init__(self, model, loss_fn, optimizer, cfg, checkpointer: Optional["Checkpointer"] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cfg = cfg
        self.checkpointer = checkpointer
        
        self.start_epoch = 0

        if self.cfg.resume_from_checkpoint and self.checkpointer:
            self._resume_training()

    def _resume_training(self):
        checkpoint = self.checkpointer.load_checkpoint()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1

    def _train_epoch(self, train_loader) -> float:
        self.model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            output = self.model(X_batch)
            loss = self.loss_fn(output, y_batch.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def _validate_epoch(self, val_loader) -> float:
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.cfg.device)
                y_batch = y_batch.to(self.cfg.device)

                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch.unsqueeze(1))

                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(val_loader)
        return avg_loss

    def train(self, train_loader, val_loader=None):
        history = {"train_loss": [], "val_loss": []}

        progress_bar = tqdm(range(self.start_epoch, self.cfg.epochs))

        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch {epoch + 1}")

            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

            if self.checkpointer:
                self.checkpointer.save_checkpoint(epoch, self.model, self.optimizer)

            metrics = {"train_loss": f"{train_loss}"}
            if val_loss is not None:
                metrics["val_loss"] = f"{val_loss}"

            progress_bar.set_postfix(metrics)

        return history