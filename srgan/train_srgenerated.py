import glob
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.models
from PIL import Image
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchmetrics import Accuracy, F1Score, AUROC, Precision
from torchvision import transforms
from torchvision.models import VGG16_Weights, ResNet18_Weights
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from train import Model, set_seed, AnimalDatSet, get_transformation

if __name__ == "__main__":
    # Load the resnet model with pre-trained weights
    model_architecture = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = Model(model=model_architecture, lr=1e-4, wd=1e-6)

    max_epochs = 100
    model_path = "../checkpoints/model_B"
    batch_size = 32
    # Set seed for reproducibility
    seed = 10
    set_seed(seed)
    # Create dataset and split it into train, and validation
    cat_dog_train_dataset = AnimalDatSet(data_root="./generated_train_images", transformation=get_transformation())
    train_loader = DataLoader(cat_dog_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    cat_dog_val_dataset = AnimalDatSet(data_root="./generated_val_images", transformation=get_transformation())
    val_loader = DataLoader(cat_dog_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # Set up logging for training progress
    csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "srgan_train.log/"))
    tb_logger = TensorBoardLogger(save_dir=os.path.join(model_path, "tb_log/"), name="resnet18")
    # Define early stopping criteria
    monitor = "val_loss"
    mode = "min"
    early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=False, mode=mode)
    # Initialize the trainer and start training
    trainer = pl.Trainer(
        default_root_dir=model_path,
        accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=[
            early_stopping,
            ModelCheckpoint(dirpath=model_path, filename="resnet18-{epoch}-{val_loss:.2f}", save_top_k=1,
                            save_weights_only=True, mode=mode, monitor=monitor),
        ],
        logger=[tb_logger, csv_logger],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
