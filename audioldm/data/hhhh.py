import sys

sys.path.append("src")
import os
import math
import pandas as pd
import zlib
import yaml
import audioldm_train.utilities.audio as Audio
from audioldm_train.utilities.tools import load_json
from audioldm_train.dataset_plugin import *
import librosa
from librosa.filters import mel as librosa_mel_fn
import threading

import random
import lmdb
from torch.utils.data import Dataset
import torch.nn.functional
import torch
from pydub import AudioSegment
import numpy as np
import torchaudio
import io
import json
from .datum_all_pb2 import Datum_all as Datum_lmdb
from .datum_mos_pb2 import Datum_mos as Datum_lmdb_mos
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class AudioDataset(Dataset):
    def __init__(
        self,
        config,
        lmdb_path,
        key_path,
        mos_path,
        lock=True
    ):
        self.config = config
        # self.lock = threading.Lock()
        """
        Dataset that manages audio recordings
        """
        self.pad_wav_start_sample = 0
        self.trim_wav = False
        self.build_setting_parameters()
        self.build_dsp()
        
        self.lmdb_path = [_.encode("utf-8") for _ in lmdb_path]
        self.lmdb_env = [lmdb.open(_, readonly=True, lock=False) for _ in self.lmdb_path]
        self.mos_txn_env = lmdb.open(mos_path, readonly=True, lock=False)
        self.key_path = [_.encode("utf-8") for id, _ in enumerate(key_path)]
        self.keys = []
        for _ in range(len(key_path)):
            with open(self.key_path[_]) as f:
                for line in f:
                    key = line.strip() 
                    self.keys.append((_, key.split()[0].encode('utf-8')))
                    # only for test !!!
                    # if _ > 20:
                    #     break
        # self.keys : [(id, key), ..., ...]

        # self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        # self.txn = self.lmdb_env.begin()
        print(f"Dataset initialize finished, dataset_length : {len(self.keys)}")
        print(f"Initialize of filter start: ")
        with open('filter_all.lst', 'r') as f:
            self.filter = {}
            for _ in f.readlines():
                self.filter[_.strip()] = 1
        print(f"Initialize of filter finished")
        #print(f"Initialize of fusion start: ")
        #with open('new_file.txt', 'r') as f:
        #    self.refined_caption = {}
        #    for _ in f.readlines():
        #        try:
        #            a, b = _.strip().split("@")
        #            b = b.strip('"\n')
        #            b = b.replace('\n', ',')
        #            self.refined_caption[a] = b    
        #        except:
        #            pass
        #print(f"Initialize of fusion finished")   

    def __getitem__(self, index):
        (

            # name of file, while we use dir of fine here
            fname,

            # wav of sr = 16000
            waveform,

            # mel
            stft,

            # log mel
            log_mel_spec,

            label_vector,

            # donot start at the begining
            random_start,

            # dict or single string which describes the wav file
            caption,
             
            # mos score for single music clip
            mos

        ) = self.feature_extraction(index)

        data = {
            "text": [caption],  # list ... dict ?
            "fname": [fname], # list
            # tensor, [batchsize, 1, samples_num]
            "waveform": "" if (waveform is None) else waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if (stft is None) else stft.float(),
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
            "label_vector": label_vector,
            "mos":mos
        }

        if data["text"] is None:
            print("Warning: The model return None on key text", fname)
            data["text"] = ""

        return data

    def __len__(self):
        return len(self.keys)

    def feature_extraction(self, index):
        if index > len(self.keys) - 1:  
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.keys) - 1)
        waveform = np.array([])
        tyu = 0
        flag = 0
        last_index = index
        while(flag == 0):
            id_, k = self.keys[index]
            try:
                if self.filter[k.decode()] == 1:
                    index = random.randint(0, len(self.keys) - 1)
                else:
                    flag = 1
            except: 
                flag = 1
        index = last_index
        while len(waveform) < 1000:
            
            id_, k = self.keys[index]
            with self.lmdb_env[id_].begin(write=False) as txn:
                cursor = txn.cursor()
                try:
                    cursor.set_key(k)
           
                    datum_tmp = Datum_lmdb()
                    datum_tmp.ParseFromString(cursor.value())
                    zobj = zlib.decompressobj()  # obj for decompressing data streams that won’t fit into memory at once.
                    decompressed_bytes = zobj.decompress(datum_tmp.wav_file)

                    # decompressed_bytes = zlib.decompress(file)
                    waveform = np.frombuffer(decompressed_bytes, dtype=np.float32)
                except:
                    tyu += 1
                    pass
            tyu += 1
            last_index = index
            index = random.randint(0, len(self.keys) - 1)
        if tyu > 1:
            print('error')
        index = last_index
        flag = 0
        val = 623787092.84794
        while (flag == 0):
            id_, k = self.keys[index]
            with self.mos_txn_env.begin(write=False) as txn:
                cursor = txn.cursor()
                try:
                    if cursor.set_key(k):
                        datum_mos = Datum_lmdb_mos()
                        datum_mos.ParseFromString(cursor.value())
                        mos = datum_mos.mos       
                    else:
                        mos = -1.0
                except :
                    mos = -1.0
            if np.random.rand() < math.exp(5.0 * mos) / val:
                flag = 1
            last_index = index
            index = random.randint(0, len(self.keys) - 1)
        index = last_index
        caption_original = datum_tmp.caption_original
        try:
            caption_generated = datum_tmp.caption_generated[0]
        except:
            caption_generated = 'None'
        assert len(caption_generated) > 1
        caption_original = caption_original.lower()
        caption_generated = caption_generated.lower()
        caption = 'music'
        if ("msd_" in k.decode()):
            caption = caption_generated if caption_original == "none" else caption_original
        elif ("audioset_" in k.decode()):
            caption = caption_generated if caption_generated != "none" else caption_original
        elif ("mtt_" in k.decode()):
            caption = caption_generated if caption_original == "none" else caption_original
        elif ("fma_" in k.decode()):
            caption = caption_generated if caption_generated != "none" else caption_original
        elif ("pixa_" in k.decode()):
            caption = caption_generated if caption_generated != "none" else caption_original
        else:
            caption = caption_original
        prefix = 'medium quality'

        miu = 3.80
        sigma = 0.20

        mos = float(mos)
        if mos > miu - sigma and mos < miu + sigma:
            prefix = "medium quality"
        elif mos >= miu + sigma:
            prefix = "high quality"
        elif mos <= miu - sigma:
            prefix = "low quality"
        else:
            print(f'mos score for key : {k.decode()} miss, please check')
        #if 'low quality' or 'quality is low' in caption:
        #    prefix = 'low quality'
        caption = prefix + ', ' + caption

        if miu - 2 * sigma <= mos < miu - sigma:
            vq_mos = 2
        elif miu - sigma <= mos < miu + sigma:
            vq_mos = 3
        elif miu + sigma <= mos < miu + 2 * sigma:
            vq_mos = 4
        elif mos >= miu + 2 * sigma:
            vq_mos = 5
        else:
            vq_mos = 1
        """
        tags = datum_tmp.tags.decode()
        caption_writing = datum_tmp.caption_writing.decode()
        caption_paraphrase = datum_tmp.caption_paraphrase.decode()
        caption_attribute_prediction = datum_tmp.caption_attribute_prediction.decode()
        caption_summary = datum_tmp.caption_summary.decode()
        """
        (
            log_mel_spec,
            stft,
            waveform,
            random_start,
        ) = self.read_audio_file(waveform, k.decode())
        fname = self.keys[index]
        # t_step = log_mel_spec.size(0)
        # waveform = torch.FloatTensor(waveform[..., : int(self.hopsize * t_step)])
        waveform = torch.FloatTensor(waveform)
        label_vector = torch.FloatTensor(np.zeros(0, dtype=np.float32))
        # finally:
        #     self.lock.release()
        # import pdb 
        # pdb.set_trace()
        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_vector,
            random_start,
            caption,
            vq_mos
        )

    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        # self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        # self.timem = self.config["preprocessing"]["mel"]["timem"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]

        # Calculate parameter derivations
        # self.waveform_sample_length = int(self.target_length * self.hopsize)

        # if (self.config["balance_sampling_weight"]):
        #     self.samples_weight = np.loadtxt(
        #         self.config["balance_sampling_weight"], delimiter=","
        #     )

        # if "train" not in self.split:
        #     self.mixup = 0.0
        #     # self.freqm = 0
        #     # self.timem = 0

    def build_dsp(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]
        self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]

        self.STFT = Audio.stft.TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )

    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        # waveform = librosa.resample(waveform, sr, self.sampling_rate)
        return waveform

        # if sr == 16000:
        #     return waveform
        # if sr == 32000 and self.sampling_rate == 16000:
        #     waveform = waveform[::2]
        #     return waveform
        # if sr == 48000 and self.sampling_rate == 16000:
        #     waveform = waveform[::3]
        #     return waveform
        # else:
        #     raise ValueError(
        #         "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
        #         % (sr, self.sampling_rate)
        #     )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length):
        waveform = torch.tensor(waveform)
        waveform = waveform.unsqueeze(0)
        waveform_length = waveform.shape[-1]
        # assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
        if waveform_length < 100:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform_length))

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        for i in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        # print(waveform)
        # import pdb 
        # pdb.set_trace()
        waveform_length = waveform.shape[-1]
        # assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
        if waveform_length < 100:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform_length))

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, file, k):
        #zobj = zlib.decompressobj()  # obj for decompressing data streams that won’t fit into memory at once.
        #decompressed_bytes = zobj.decompress(file)

        # decompressed_bytes = zlib.decompress(file)
        #waveform = np.frombuffer(decompressed_bytes, dtype=np.float32)
        waveform = file
        # # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        # if "msd" in k or "fma" in k:
        #     try:
        #         waveform = torch.tensor([(np.array(file.get_array_of_samples(array_type_override='i')) / 2147483648)], dtype=torch.float32)
        #     except:
        #         waveform = torch.tensor([(np.array(file.get_array_of_samples(array_type_override='h')) / 32768)], dtype=torch.float32)
        # else:
        #     waveform = torch.tensor([(np.array(file.get_array_of_samples(array_type_override='h')) / 32768)], dtype=torch.float32)
        # # else:
        #     # raise AttributeError
        
        # waveform = torch.tensor([(np.array(file.get_array_of_samples(array_type_override='h')) / 32768)], dtype=torch.float32)
        # import pdb 
        # pdb.set_trace()
        sr = 16000
        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )
        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        if self.trim_wav:
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, random_start

    def mix_two_waveforms(self, waveform1, waveform2):
        mix_lambda = np.random.beta(5, 5)
        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        return self.normalize_wav(mix_waveform), mix_lambda

    def read_audio_file(self, file, k):
        # target_length = int(self.sampling_rate * self.duration)
        # import pdb 
        # pdb.set_trace()
        # print(type(file))
        waveform, random_start = self.read_wav_file(file, k)
        
        # log_mel_spec, stft = self.wav_feature_extraction_torchaudio(waveform) # this line is faster, but this implementation is not aligned with HiFi-GAN
        log_mel_spec, stft = self.wav_feature_extraction(waveform)

        return log_mel_spec, stft, waveform, random_start

    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))
        # import pdb 
        # pdb.set_trace()
        if self.mel_fmax not in self.mel_basis:
            # import pdb 
            # pdb.set_trace()
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)
        # import pdb 
        # pdb.set_trace()
        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def _read_datum_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        random_index = torch.randint(0, len(caption_keys), (1,))[0].item()
        return datum[caption_keys[random_index]]

    def _is_contain_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        return len(caption_keys) > 0

    def label_indices_to_text(self, datum, label_indices):
        if self._is_contain_caption(datum):
            return self._read_datum_caption(datum)
        elif "label" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]
            # description_header = "This audio contains the sound of "
            description_header = ""
            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return description_header + labels
        else:
            return ""  # TODO, if both label and caption are not provided, return empty string

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec


class AudioDataset_infer(Dataset):
    def __init__(
        self,
        config,
        caption_list,
        lock=True
    ):
        self.config = config
        # self.lock = threading.Lock()
        """
        Dataset that manage caption writings
        """
        self.captions = []
        with open(caption_list, 'r') as f:
            for _ ,line in enumerate(f):
                key = line.strip() 
                self.captions.append(key.split()[0])
        self.duration = self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.sampling_rate = self.config["variables"]["sampling_rate"]
        self.target_length = int(self.sampling_rate * self.duration)
        self.waveform = torch.zeros((1, self.target_length))

    def __getitem__(self, index):
        
        fname = [f"sample_{index}"]
        data = {
            "text": [self.captions[index]],  # list ... dict ?
            "fname": fname, # list
            # tensor, [batchsize, 1, samples_num]
            "waveform": "",
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "",
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "",
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": 0,
            "label_vector": torch.FloatTensor(np.zeros(0, dtype=np.float32)),
            "mos":mos
        }

        if data["text"] is None:
            print("Warning: The model return None on key text", fname)
            data["text"] = ""

        return data

    def __len__(self):
        return len(self.captions)

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(0)

    def write_json(my_dict, fname):
        # print("Save json file at "+fname)
        json_str = json.dumps(my_dict)
        with open(fname, "w") as json_file:
            json_file.write(json_str)

    def load_json(fname):
        with open(fname, "r") as f:
            data = json.load(f)
            return data

    config = yaml.load(
        open(
            "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/config/vae_48k_256/ds_8_kl_1.0_ch_16.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    add_ons = config["data"]["dataloader_add_ons"]

    # load_json(data)
    dataset = AudioDataset(
        config=config, split="train", waveform_only=False, add_ons=add_ons
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    # for cnt, each in tqdm(enumerate(loader)):
        # print(each["waveform"].size(), each["log_mel_spec"].size())
        # print(each['freq_energy_percentile'])
        # import ipdb

        # ipdb.set_trace()
        # pass

