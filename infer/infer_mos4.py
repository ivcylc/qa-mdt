import shutil
import os

import argparse
import yaml
import torch

from audioldm_train.utilities.data.dataset_original_mos4 import AudioDataset as AudioDataset

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from audioldm_train.utilities.tools import get_restore_step
from audioldm_train.utilities.model_util import instantiate_from_config
from audioldm_train.utilities.tools import build_dataset_json_from_list

def infer(dataset_key, configs, config_yaml_path, exp_group_name, exp_name):
    
    seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])

    log_path = configs["log_directory"]
    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []
    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_key
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)
# /disk1/changli/jiqun_training_checkpoints/checkpoints/
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

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    guidance_scale = configs["model"]["params"]["evaluation_params"][
        "unconditional_guidance_scale"
    ]
    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"][
        "ddim_sampling_steps"
    ]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"][
        "n_candidates_per_samples"
    ]
    # resume_from_checkpoint = ""
    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"],strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-l",
        "--list_inference",
        type=str,
        required=False,
        help="The filelist that contain captions (and optionally filenames)",
    )

    parser.add_argument(
        "-reload_from_ckpt",
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="the checkpoint path for the model",
    )

    args = parser.parse_args()
    # import pdb
    # pdb.set_trace()
    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml
    dataset_key = build_dataset_json_from_list(args.list_inference)
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    infer(dataset_key, config_yaml, config_yaml_path, exp_group_name, exp_name)