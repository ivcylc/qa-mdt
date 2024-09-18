# SOTA awesome text to music generation (TTM) model QA-MDT

**Official Pytorch Implementation**

**without any fancy design, just a quality injection, and enjoy your beautiful music**

<a href="https://arxiv.org/pdf/2405.15863"><img src="https://img.shields.io/static/v1?label=Paper&message=arxiv.2405&color=red&logo=arxiv"></a> &ensp;
<a href="https://qa-mdt.github.io/"><img src="https://img.shields.io/static/v1?label=Demo&message=QA-MDT&color=yellow&logo=github.io"></a> &ensp;
<a href="https://huggingface.co/lichang0928/QA-MDT"><img src="https://img.shields.io/static/v1?label=ckpts&message=QA-MDT&color=black&logo=huggingface.co"></a> &ensp;

For chinese users, you can also download your checkpoint through following link:

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
```

Before training, you need to download extra ckpts needed in ./audioldm_train/config/mos_as_token/qa_mdt.yaml and offset_pretrained_checkpoints.json

Noted that: All above checkpoints can be downloaded from:

[flan-t5-large](https://huggingface.co/google/flan-t5-large)

[clap_music](https://huggingface.co/lukewys/laion_clap/blob/main/music_speech_audioset_epoch_15_esc_89.98.pt)

[roberta-base](https://huggingface.co/FacebookAI/roberta-base)

[others](https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS/view)


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

## Contact
This is the first time I open source such a project, the code, the organization, the open source may not be perfect.
If you have any questions, feel free to contact me via, and I'm looking forward to any suggestions:

- **Email**: [lc_lca@mail.ustc.edu.cn](mailto:lc_lca@mail.ustc.edu.cn)
- **WeChat**: 19524292801

## Citation

If you find this project useful, please consider citing:

```bash
@article{li2024quality,
  title={Quality-aware Masked Diffusion Transformer for Enhanced Music Generation},
  author={Li, Chang and Wang, Ruoyu and Liu, Lijuan and Du, Jun and Sun, Yixuan and Guo, Zilu and Zhang, Zhenrong and Jiang, Yuan},
  journal={arXiv preprint arXiv:2405.15863},
  year={2024}
}
