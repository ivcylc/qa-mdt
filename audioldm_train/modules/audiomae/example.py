import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import to_2tuple
import models_vit
from audiovisual_dataset import AudioVisualDataset, collate_fn
from torch.utils.data import DataLoader
from util.stat import calculate_stats
from tqdm import tqdm
from AudioMAE import AudioMAE

if __name__ == "__main__":
    device = "cuda"
    dataset = AudioVisualDataset(
        datafiles=[
            "/mnt/bn/data-xubo/dataset/audioset_videos/datafiles/audioset_eval.json"
        ],
        # disable SpecAug during evaluation
        freqm=0,
        timem=0,
        return_label=True,
    )

    model = AudioMAE().to(device)
    model.eval()

    outputs = []
    targets = []

    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=8, shuffle=False, collate_fn=collate_fn
    )

    print("Start evaluation on AudioSet ...")
    with torch.no_grad():
        for data in tqdm(dataloader):
            fbank = data["fbank"]  # [B, 1, T, F]
            fbank = fbank.to(device)
            output = model(fbank, mask_t_prob=0.0, mask_f_prob=0.0)
            target = data["labels"]
            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    stats = calculate_stats(outputs, targets)

    AP = [stat["AP"] for stat in stats]
    mAP = np.mean([stat["AP"] for stat in stats])
    print("Done ... mAP: {:.6f}".format(mAP))

    # mAP: 0.463003
