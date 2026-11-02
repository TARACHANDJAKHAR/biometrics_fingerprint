"""
Training Pipeline for DeepPrint-Inspired Model.

Supports two-stage fine-tuning (frozen backbone warmup -> unfrozen fine-tuning),
early stopping, learning rate scheduling, model checkpointing, metric history logging,
and live real-time progress bars with tqdm.
"""

import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """
    Manages model training, validation, checkpointing, and metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": []
        }

    def train_epoch(self, optimizer: torch.optim.Optimizer, desc: str = "Training") -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=desc, leave=True)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            running_loss = total_loss / max(total, 1)
            running_acc = (correct / max(total, 1)) * 100.0
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.1f}%"
            })

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = (correct / max(total, 1)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, desc: str = "Validation") -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=desc, leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix({
                "loss": f"{total_loss / max(total, 1):.4f}",
                "acc": f"{(correct / max(total, 1)) * 100.0:.1f}%"
            })

        val_loss = total_loss / max(total, 1)
        val_acc = (correct / max(total, 1)) * 100.0
        return val_loss, val_acc

    def fit(
        self,
        epochs_stage1: int = 5,
        epochs_stage2: int = 10,
        lr_stage1: float = 1e-3,
        lr_stage2: float = 1e-4,
        patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Executes two-stage transfer learning with live progress bars:
        Stage 1: Frozen backbone (train linear projection + classification head).
        Stage 2: Unfrozen deep layers (fine-tune end-to-end representation).
        """
        best_val_loss = float("inf")
        patience_counter = 0

        # === STAGE 1: Warmup with frozen backbone ===
        print(f"\n{'='*20} STAGE 1: Warmup (Backbone Frozen) {'='*20}")
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        trainable_params_s1 = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params_s1, lr=lr_stage1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage1)

        for epoch in range(1, epochs_stage1 + 1):
            t0 = time.time()
            desc = f"Stage 1 [Epoch {epoch}/{epochs_stage1}]"
            train_loss, train_acc = self.train_epoch(optimizer, desc=desc)
            val_loss, val_acc = self.evaluate(desc=f"Val {epoch}/{epochs_stage1}")
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            print(f"--> Stage 1 | Epoch [{epoch:02d}/{epochs_stage1:02d}] ({elapsed:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model_stage1.pth")

        # === STAGE 2: End-to-End Fine-Tuning ===
        print(f"\n{'='*20} STAGE 2: Fine-Tuning (Deep Layers Unfrozen) {'='*20}")
        if hasattr(self.model, "unfreeze_deep_layers"):
            self.model.unfreeze_deep_layers()
        elif hasattr(self.model, "unfreeze_all"):
            self.model.unfreeze_all()

        trainable_params_s2 = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params_s2, lr=lr_stage2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)

        for epoch in range(1, epochs_stage2 + 1):
            t0 = time.time()
            desc = f"Stage 2 [Epoch {epoch}/{epochs_stage2}]"
            train_loss, train_acc = self.train_epoch(optimizer, desc=desc)
            val_loss, val_acc = self.evaluate(desc=f"Val {epoch}/{epochs_stage2}")
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            print(f"--> Stage 2 | Epoch [{epoch:02d}/{epochs_stage2:02d}] ({elapsed:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"    * Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] No validation improvement for {patience} epochs.")
                    break

        self.save_checkpoint("last_model.pth")
        self.plot_training_curves()
        return self.history

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history
        }, path)

    def plot_training_curves(self, save_filename: str = "training_curves.png"):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", marker='o')
        axes[0].plot(epochs, self.history["val_loss"], label="Val Loss", marker='s')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        axes[1].plot(epochs, self.history["train_acc"], label="Train Acc (%)", marker='o')
        axes[1].plot(epochs, self.history["val_acc"], label="Val Acc (%)", marker='s')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training curves saved to: {save_path}")
