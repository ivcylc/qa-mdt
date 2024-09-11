# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import sys

sys.path.append("src")

import os
import wandb

import argparse
import yaml
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from audioldm_train.utilities.data.dataset import AudioDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from audioldm_train.modules.latent_encoder.autoencoder import AutoencoderKL
from pytorch_lightning.callbacks import ModelCheckpoint
from audioldm_train.utilities.tools import get_restore_step


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


def main(configs, exp_group_name, exp_name):
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])
    batch_size = config_yaml["model"]["params"]["batchsize"]
    log_path = config_yaml["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = AudioDataset(config_yaml, split="train", add_ons=dataloader_add_ons)

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = AudioDataset(config_yaml, split="val", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )

    model = AutoencoderKL(
        ddconfig=config_yaml["model"]["params"]["ddconfig"],
        lossconfig=config_yaml["model"]["params"]["lossconfig"],
        embed_dim=config_yaml["model"]["params"]["embed_dim"],
        image_key=config_yaml["model"]["params"]["image_key"],
        base_learning_rate=config_yaml["model"]["base_learning_rate"],
        subband=config_yaml["model"]["params"]["subband"],
        sampling_rate=config_yaml["preprocessing"]["audio"]["sampling_rate"],
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-{global_step:.0f}",
        every_n_train_steps=5000,
        save_top_k=config_yaml["step"]["save_top_k"],
        auto_insert_metric_name=False,
        save_last=True,
    )

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    model.set_log_dir(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)

    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=config_yaml["project"],
        config=config_yaml,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        limit_val_batches=100,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=True),
        val_check_interval=2000,
    )

    # TRAINING
    trainer.fit(model, loader, val_loader, ckpt_path=resume_from_checkpoint)

    # EVALUTION
    # trainer.test(model, test_loader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--autoencoder_config",
        type=str,
        required=True,
        help="path to autoencoder config .yam",
    )

    args = parser.parse_args()

    config_yaml = args.autoencoder_config
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml = os.path.join(config_yaml)

    config_yaml = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

    main(config_yaml, exp_group_name, exp_name)
