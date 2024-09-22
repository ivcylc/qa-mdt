# Awesome text to music generation (TTM): QA-MDT (Openmusic)

## Official Pytorch Implementation
## I recommend anyone to listen to our demo, even under the clutter of tabs in Musiccaps, we still perform well

**We have to admit that the Unet architecture still has some probability advantage in subjective musicality, but this is not measured in the metric.
And, we did have some models that were better on the metric, or trained for longer, but we observed that the models generally became less musicality after training too long, so we picked a model that was moderate on the metric as an open source sample. If you need more models (extreme metric pursuit or extreme musically pursuit, please contact me)**

**without any fancy design, just a quality injection, and enjoy your beautiful music**

<a href="https://arxiv.org/pdf/2405.15863"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv.2405&color=red&logo=arXiv"></a> &ensp;
<a href="https://qa-mdt.github.io/"><img src="https://img.shields.io/static/v1?label=Demo&message=QA-MDT&color=black&logo=github.io"></a> &ensp;
<a href="https://huggingface.co/lichang0928/QA-MDT"><img src="https://img.shields.io/static/v1?label=ckpts&message=huggingface&color=yellow&logo=huggingface.co"></a> &ensp;

Down the main checkpoint of our QA-MDT model from [https://huggingface.co/lichang0928/QA-MDT](https://huggingface.co/lichang0928/QA-MDT)

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

### How to Prepare for Training or Fine-tuning

Our model is already well-pretrained. If you wish to retrain or fine-tune it, you can choose to use or not use our QA strategy. We offer several training strategies:

- **MDT w.o quality token**: `PixArt_MDT`
- **MDT with quality token**: `Pixart_MDT_MOS_AS_TOKEN`
- **DiT**: `PixArt_Slow`
- **U-net w / w.o quality prefix**: `you can just follow AudioLDM and make your dataset as illustrated in our paper (method part)`

To train or fine-tune, simply change `"Your_Class"` in `audioldm_train.modules.diffusionmodules.PixArt.Your_Class` in our [config file](https://github.com/ivcylc/qa-mdt/blob/main/audioldm_train/config/mos_as_token/qa_mdt.yaml).

you can also try modifying the patch size, overlap size for your best performance and computing resources trade off (see our Appendix in arXiv paper)

#### How to Prepare Your Dataset for Training or Fine-tuning

We use the **LMDB** dataset format for training. You can modify the dataloader according to your own training needs.

If you'd like to follow our process (though we don't recommend it, as it can be complex), here's how you can create a toy LMDB dataset:

1. **Create a Proto File**

   First, create a file named `datum_all.proto` with the following content:

   ```proto
   syntax = "proto2";

   message Datum_all {
     repeated float wav_file = 1;
     required string caption_original = 2;
     repeated string caption_generated = 3;
     required float mos = 4;
   }
2. **Generate Python Bindings**

  Run the following command in your terminal to generate Python bindings:

  ```bash
  protoc --python_out=./ datum_all.proto
  ```

  This will create a file called **datum_all_pb2.py**. We have also provided this file in our datasets folder, and you can check if it matches the one you generated. **Never attempt to modify this file, as doing so could cause errors.**
  
3. **Code for Preparing a toy LMDB Dataset**

  The following Python script demonstrates how to prepare your dataset in the LMDB format:
  
  ```python
  import torch
  import os
  import lmdb
  import time
  import numpy as np
  import librosa
  import os
  import soundfile as sf
  import io
  
  from datum_all_pb2 import Datum_all as Datum_out
  
  device = 'cpu'
  count = 0
  total_hours = 0
  
  # Define paths
  lmdb_file = '/disk1/changli/toy_lmdb'
  toy_path = '/disk1/changli/audioset'
  lmdb_key = os.path.join(lmdb_file, 'data_key.key')
  
  # Open LMDB environment
  env = lmdb.open(lmdb_file, map_size=1e12)
  txn = env.begin(write=True)
  final_keys = []
  
  def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs):
      """Load and resample audio using librosa."""
      src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
      return src
  
  start_time = time.time()
  
  # Walk through the dataset directory
  for root, _, files in os.walk(toy_path):
      for file in files:
          audio_path = os.path.join(root, file)
          key_tmp = audio_path.replace('/', '_')
          audio = _resample_load_librosa(audio_path, 16000, True)
          
          # Create a new Datum object
          datum = Datum_out()
          datum.wav_file.extend(audio)
          datum.caption_original = 'audio'.encode()
          datum.caption_generated.append('audio'.encode())
          datum.mos = -1
  
          # Write to LMDB
          txn.put(key_tmp.encode(), datum.SerializeToString())
          final_keys.append(key_tmp)
  
          count += 1
          total_hours += 1.00 / 60 / 10
  
          if count % 1 == 0:
              elapsed_time = time.time() - start_time
              print(f'{count} files written, time: {elapsed_time:.2f}s')
              txn.commit()
              txn = env.begin(write=True)
  
  # Finalize transaction
  try:
      total_time = time.time() - start_time
      print(f'Packing completed: {count} files written, total_hours: {total_hours:.2f}, time: {total_time:.2f}s')
      txn.commit()
  except:
      pass
  
  env.close()
  
  # Save the LMDB keys
  with open(lmdb_key, 'w') as f:
      for key in final_keys:
          f.write(key + '\n')
  ```

4. **Input your generated lmdb path and its corresponding key file path into the config**

5. **Start your training**

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
If you have any questions about our model, code and datasets, feel free to contact me via below links, and I'm looking forward to any suggestions:

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
```
