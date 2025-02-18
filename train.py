import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path

from data.dataset import ChessDataset
from data.transforms import ChessTransforms
from models.model import create_model
from utils.device import get_device


class DLTrainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.best_val_acc = 0
        self.patience_counter = 0

        # Create save directory
        self.save_dir = Path("saved_models")
        self.save_dir.mkdir(exist_ok=True)

    def setup_data(self):
        # Load DataFrame
        df = pd.read_csv(self.config.data.csv_path)

        # Create transforms
        train_transform = ChessTransforms(self.config, is_train=True)
        val_transform = ChessTransforms(self.config, is_train=False)

        # Split data using config value
        train_df = df.sample(frac=self.config.data.train_split_frac, random_state=42)
        val_df = df.drop(train_df.index)

        # Create datasets
        train_dataset = ChessDataset(train_df, transform=train_transform)
        val_dataset = ChessDataset(val_df, transform=val_transform)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

    def setup_model(self):
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        accuracy = 100.0 * correct / total

        # Save best model
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return total_loss / len(self.val_loader), accuracy

    def train(self):
        self.setup_data()
        self.setup_model()

        print(f"Training on device: {self.device}")

        for epoch in range(self.config.training.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch: {epoch+1}/{self.config.training.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)

            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print("Early stopping triggered")
                break


def main():
    # Load config using OmegaConf
    config = OmegaConf.load("config/train_config.yaml")

    # Create trainer and start training
    trainer = DLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
