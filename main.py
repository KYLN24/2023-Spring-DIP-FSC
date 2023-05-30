import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import FSCData
from model import LitModule

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 1
LR = 1e-4
LOG_FREQ = 2

torch.set_float32_matmul_precision("high")

model = LitModule(
    optimizer="adamw",
    scheduler="step",
    learning_rate=LR,
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    dmap_scale=100,
)

train_set = FSCData("./data", method="train")
train_loader = DataLoader(
    train_set,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=32,
    drop_last=True,
)

val_set = FSCData("./data", method="val")
val_loader = DataLoader(
    val_set,
    batch_size=VAL_BATCH_SIZE,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=64,
)

wandb_logger = WandbLogger(project="fsc")
early_stop_callback = EarlyStopping(monitor="val/mae", patience=10, mode="min")

model_checkpoint_callback = ModelCheckpoint(monitor="val/mae", mode="min")

trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="cuda",
    devices=[0, 1, 2],
    logger=wandb_logger,
    log_every_n_steps=LOG_FREQ,
    benchmark=True,
    callbacks=[early_stop_callback, model_checkpoint_callback],
    check_val_every_n_epoch=5,
    num_sanity_val_steps=0,
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

test_set = FSCData("./data", method="test")
test_loader = DataLoader(
    test_set,
    batch_size=1,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=64,
)

trainer.test(model, test_loader)
