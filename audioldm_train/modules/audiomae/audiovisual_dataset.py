import json
import random
from tqdm import tqdm
import torch
import decord

decord.bridge.set_bridge("torch")
import torchaudio
from math import ceil
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class AudioVisualDataset(Dataset):
    """Can sample data from audio-visual databases
    Params:
    min_video_frames: used to drop short video clips
    video_resize: resize for CLIP processing
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audiovisual clip to be sampled
    num_sample_frames: number of image frames to be uniformly sampled from video
    """

    def __init__(
        self,
        datafiles=[
            "/mnt/bn/data-xubo/dataset/audioset_videos/datafiles/audioset_balanced_train.json",
        ],
        min_video_frames=30,
        video_resize=[224, 224],
        sampling_rate=16000,
        sample_av_clip=True,
        max_clip_len=10,
        num_sample_frames=10,
        # hyparameters used for SpecAug
        freqm=48,
        timem=192,
        return_label=False,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, "r") as fp:
                data_json = json.load(fp)["data"]
                all_data_json.extend(data_json)

        # drop short video clips
        self.all_data_json = [
            data
            for data in all_data_json
            if int(data["video_shape"][0]) >= min_video_frames
        ]

        self.max_clip_len = max_clip_len
        self.video_resize = video_resize
        self.sampling_rate = sampling_rate
        self.sample_av_clip = sample_av_clip
        self.num_sample_frames = num_sample_frames
        self.corresponding_audio_len = self.sampling_rate * self.max_clip_len

        # hyparameters used for AudioMAE
        self.freqm = freqm
        self.timem = timem
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974
        self.melbins = 128
        self.TARGET_LEN = 1024

        self.return_label = return_label
        if self.return_label:
            self.audioset_label2idx = self._prepare_audioset()

    def __len__(self):
        return len(self.all_data_json)

    def _read_audio_video(self, index):
        try:
            video_path = self.all_data_json[index]["mp4"]
            # read audio
            ar = decord.AudioReader(
                video_path, sample_rate=self.sampling_rate, mono=True
            )
            # read video frames
            vr = decord.VideoReader(
                video_path,
                height=self.video_resize[0],
                width=self.video_resize[1],
            )

            labels = self.all_data_json[index]["labels"]
            return vr, ar, labels

        except Exception as e:
            print(f"error: {e} occurs, when loading {video_path}")
            random_index = random.randint(0, len(self.all_data_json) - 1)
            return self._read_audio_video(index=random_index)

    def _prepare_audioset(self):
        df1 = pd.read_csv(
            "/mnt/bn/lqhaoheliu/datasets/audioset/metadata/class_labels_indices.csv",
            delimiter=",",
            skiprows=0,
        )
        label_set = df1.to_numpy()
        code2id = {}
        for i in range(len(label_set)):
            code2id[label_set[i][1]] = label_set[i][0]
        return code2id

    def __getitem__(self, index):
        # read audio and video
        vr, ar, labels = self._read_audio_video(index)

        # create a audio tensor
        audio_data = ar[:]  # [1, samples]
        audio_len = audio_data.shape[1] / self.sampling_rate
        audio_data = audio_data.squeeze(0)  # [samples]

        # create a video tensor
        full_vid_length = len(vr)
        video_rate = ceil(vr.get_avg_fps())
        samples_per_frame = float(self.sampling_rate) / video_rate
        start_frame = 0

        # sample video clip
        if audio_len > self.max_clip_len and self.sample_av_clip:
            start_frame = random.randint(
                0, max(full_vid_length - video_rate * self.max_clip_len, 0)
            )
        end_frame = min(start_frame + video_rate * self.max_clip_len, full_vid_length)
        video_data = vr.get_batch(range(start_frame, end_frame))

        # sample audio clip
        if audio_len > self.max_clip_len and self.sample_av_clip:
            # corresponding_audio_len = int(video_data.size()[0] * samples_per_frame)
            corresponding_audio_start = int(start_frame * samples_per_frame)
            audio_data = audio_data[corresponding_audio_start:]

        # cut or pad audio clip with respect to the sampled video clip
        if audio_data.shape[0] < self.corresponding_audio_len:
            zero_data = torch.zeros(self.corresponding_audio_len)
            zero_data[: audio_data.shape[0]] = audio_data
            audio_data = zero_data
        elif audio_data.shape[0] > self.corresponding_audio_len:
            audio_data = audio_data[: self.corresponding_audio_len]

        # uniformly sample image frames from video [tentative solution]
        interval = video_data.shape[0] // self.num_sample_frames
        video_data = video_data[::interval][: self.num_sample_frames]

        assert (
            video_data.shape[0] == self.num_sample_frames
        ), f"number of sampled image frames is {video_data.shape[0]}"

        assert (
            audio_data.shape[0] == self.corresponding_audio_len
        ), f"number of audio samples is {audio_data.shape[0]}"

        # video transformation
        video_data = video_data / 255.0
        video_data = video_data.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # calculate mel fbank of waveform for audio encoder
        audio_data = audio_data.unsqueeze(0)  # [1, samples]
        audio_data = audio_data - audio_data.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=10,
        )
        # cut and pad
        n_frames = fbank.shape[0]
        p = self.TARGET_LEN - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.TARGET_LEN, :]

        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)  # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        fbank = fbank.unsqueeze(0)

        if self.return_label:
            # get audioset lebel indexes
            label_indices = np.zeros(527)

            for label_str in labels.split(","):
                label_indices[int(self.audioset_label2idx[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

            data_dict = {
                "labels": label_indices,
                "images": video_data,
                "fbank": fbank,
                # 'modality': 'audio_visual'
            }

        else:
            data_dict = {
                "images": video_data,
                "fbank": fbank,
                # 'modality': 'audio_visual'
            }

        return data_dict


def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (channels_num, segment_samples),
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            {'vocals': (channels_num, segment_samples),
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, channels_num, segment_samples),
            'accompaniment': (batch_size, channels_num, segment_samples),
            'mixture': (batch_size, channels_num, segment_samples)
            }
    """

    data_dict = {}
    for key in list_data_dict[0].keys():
        # for key in ['waveform']:
        # try:
        data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        # except:
        #     from IPython import embed; embed(using=False); os._exit(0)

        data_dict[key] = torch.stack(data_dict[key])

    return data_dict
