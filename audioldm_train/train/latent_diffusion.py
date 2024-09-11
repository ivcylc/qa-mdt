import sys

# sys.path.append("src")
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import yaml
import torch


from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy


from audioldm_train.modules.latent_diffusion.ddpm import LatentDiffusion


from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from audioldm_train.utilities.tools import (
    listdir_nohidden,
    get_restore_step,
    copy_test_subset_data,
)
import wandb
from audioldm_train.utilities.model_util import instantiate_from_config
import logging

logging.basicConfig(level=logging.WARNING)



def convert_path(path):
    parts = path.decode().split("/")[-4:]
    base = ""
    result = "/".join(parts)

def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation):
    print("MAIN START")
    # cpth = "/train20/intern/permanent/changli7/dataset_ptm/test_dataset/dataset/audioset/zip_audios/unbalanced_train_segments/unbalanced_train_segments_part9/Y7fmOlUlwoNg.wav"
    # convert_path(cpth)
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(1234)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        )  # highest, high, medium

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    train_lmdb_path = configs["train_path"]["train_lmdb_path"]
    train_key_path = [_ + '/data_key.key' for _ in train_lmdb_path]

    val_lmdb_path = configs["val_path"]["val_lmdb_path"]
    val_key_path = configs["val_path"]["val_key_path"]
    

    #try:
    mos_path = configs["mos_path"]
    from audioldm_train.utilities.data.hhhh import AudioDataset
    dataset = AudioDataset(config=configs, lmdb_path=train_lmdb_path, key_path=train_key_path, mos_path=mos_path)
    

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
    )
    
   

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )
    try:
        val_dataset = AudioDataset(config=configs, lmdb_path=val_lmdb_path, key_path=val_key_path, mos_path=mos_path)
    except:
        val_dataset = AudioDataset(config=configs, lmdb_path=val_lmdb_path, key_path=val_key_path)

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
    )

    # Copy test data
    import os 
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        "tmp",
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    # copy to test:
    # import pdb
    # pdb.set_trace()
    # for i in range(len(val_dataset.keys)):
    #     key_tmp = val_dataset.keys[i].decode()
    #     cmd = "cp {} {}".format(key_tmp, os.path.join(test_data_subset_folder))
    #     os.system(cmd)

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    try:
        limit_val_batches = configs["step"]["limit_val_batches"]
    except:
        limit_val_batches = None
    
    
    validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]
    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["step"]["save_top_k"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)
    # shutil.copy(config_yaml_path, wandb_path)

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
    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)
    
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=2.0,callbacks=[checkpoint_callback],num_nodes=1,
    )

    trainer.fit(latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint)

    ################################################################################################################
    # if(resume_from_checkpoint is not None):
    #     ckpt = torch.load(resume_from_checkpoint)["state_dict"]

    #     key_not_in_model_state_dict = []
    #     size_mismatch_keys = []
    #     state_dict = latent_diffusion.state_dict()
    #     print("Filtering key for reloading:", resume_from_checkpoint)
    #     print("State dict key size:", len(list(state_dict.keys())), len(list(ckpt.keys())))
    #     for key in tqdm(list(ckpt.keys())):
    #         if(key not in state_dict.keys()):
    #             key_not_in_model_state_dict.append(key)
    #             del ckpt[key]
    #             continue
    #         if(state_dict[key].size() != ckpt[key].size()):
    #             del ckpt[key]
    #             size_mismatch_keys.append(key)

    #     if(len(key_not_in_model_state_dict) != 0 or len(size_mismatch_keys) != 0):
    #         print("⛳", end=" ")

    #     print("==> Warning: The following key in the checkpoint is not presented in the model:", key_not_in_model_state_dict)
    #     print("==> Warning: These keys have different size between checkpoint and current model: ", size_mismatch_keys)

    #     latent_diffusion.load_state_dict(ckpt, strict=False)

    # if(perform_validation):
    #     trainer.validate(latent_diffusion, val_loader)

    # trainer.fit(latent_diffusion, loader, val_loader)
    ################################################################################################################


if __name__ == "__main__":
    print("ok")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )
    parser.add_argument("--val", action="store_true")
    args = parser.parse_args()
    perform_validation = args.val
    assert torch.cuda.is_available(), "CUDA is not available"
    config_yaml = args.config_yaml
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))
    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation)
