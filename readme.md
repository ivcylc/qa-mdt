# SOTA awesome text to music generation (TTM)
# QA-MDT Implementation

checkpoint is provisionally provided, we will update more and debug(potential) soon
链接: https://pan.baidu.com/s/1pkLnQhbNeFjKRadXUy_7Iw?pwd=v9dd 提取码: v9dd
--来自百度网盘超级会员v8的分享

## Overview

This repository provides an implementation of QA-MDT, integrating state-of-the-art models for music generation. The code and methods are based on the following repositories:

- [AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning)
- [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- [MDT](https://github.com/sail-sg/MDT)
- [AudioMAE](https://github.com/facebookresearch/AudioMAE)
- [Open-Sora](https://github.com/hpcaitech/Open-Sora)

## Requirements
```bash
Python 3.10
qamdt.yaml
Downloaded all checkpoints needed in ./audioldm_train/config/mos_as_token/qa_mdt.yaml and offset_pretrained_checkpoints.json
```

## Training

```bash
sh run.sh
```

## Inference

```bash
sh infer/infer.sh
# you may change the infer.sh for witch quality level you want to infer
# defaultly, it should be set to 5 which represent highest quality
# Additionly, it may be useful to change the prompt with text prefix "high quality", 
# which match the training process and may further improve performance
```

