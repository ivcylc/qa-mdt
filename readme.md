# SOTA awesome text to music generation (TTM) model QA-MDT

**Official Pytorch Implementation**

<a href="https://arxiv.org/pdf/2405.15863"><img src="https://img.shields.io/static/v1?label=Paper&message=QA-MDT&color=red&logo=arxiv"></a> &ensp;
<a href="[qa-mdt.github.io](https://qa-mdt.github.io/)"><img src="https://img.shields.io/static/v1?label=Demo&message=QA-MDT&color=yellow&logo=github.io"></a> &ensp;

checkpoint is provisionally provided, we will update more and debug(potential) soon

https://pan.baidu.com/s/1pkLnQhbNeFjKRadXUy_7Iw?pwd=v9dd  

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

