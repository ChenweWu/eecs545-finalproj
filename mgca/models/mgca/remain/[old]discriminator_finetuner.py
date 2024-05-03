import datetime
import os
from argparse import ArgumentParser

import torch
import torchvision.models as models
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.mgca.mgca_module import MGCA
from mgca.models.ssl_finetuner import SSLFineTuner_demo, SSLFineTuner_age, SSLFineTuner_disease
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
        
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=0.01)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 5

    seed_everything(args.seed)


    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                                DataTransforms, 1.0,
                                args.batch_size, args.num_workers, 512)

    #import torchxrayvision as xrv
    #discriminator = xrv.models.ResNet(weights="resnet50-res512-all")
    
    discriminator = models.resnet50(pretrained=True)
    discriminator.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    discriminator.fc = torch.nn.Identity()

    #from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
    #import torch


    #discriminator = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-huge-22k-512")
    #discriminator.fc = torch.nn.Identity()

    args.backbone = discriminator
    args.in_features = 2048
    args.num_classes = 16
    args.multilabel = False

    # finetune
    tuner = SSLFineTuner_demo(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/mgca_finetune/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data/wandb")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="mgca_finetune",
        save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
