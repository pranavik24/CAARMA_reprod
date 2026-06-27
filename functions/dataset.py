import collections
import os
import random

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf

import torchaudio

from .voxceleb_split import load_training_dataframe

def load_audio(filename, second=3):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform.squeeze(0)  # Remove channel dimension if mono

    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform.clone()

    length = int(sr * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = torch.nn.functional.pad(waveform, (0, shortage), mode='reflect')
    else:
        start = int(random.random() * (audio_length - length))
        waveform = waveform[start:start + length]

    return waveform.clone()


class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, do_augmentation=False, augmentation=None, **kwargs):
        self.second = second

        df = load_training_dataframe(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.speaker_ids = df["utt_spk_id"].values if "utt_spk_id" in df.columns else None
        self.genders = df["gender"].values if "gender" in df.columns else None
        self.nationalities = df["nationality"].values if "nationality" in df.columns else None
        shuffle_inputs = [self.labels, self.paths]
        optional_metadata = [self.speaker_ids, self.genders, self.nationalities]
        shuffle_inputs.extend(metadata for metadata in optional_metadata if metadata is not None)
        shuffled = shuffle(*shuffle_inputs)
        self.labels, self.paths = shuffled[:2]
        metadata_index = 2
        if self.speaker_ids is not None:
            self.speaker_ids = shuffled[metadata_index]
            metadata_index += 1
        if self.genders is not None:
            self.genders = shuffled[metadata_index]
            metadata_index += 1
        if self.nationalities is not None:
            self.nationalities = shuffled[metadata_index]
        self.augmentation = augmentation
        self.do_augmentation = do_augmentation

        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_1 = load_audio(self.paths[index], self.second)
        waveform_length = waveform_1.shape[-1]
        if self.do_augmentation:
            waveform_1 = self.augmentation(waveform_1)
        
        sample = {
        'waveform':  waveform_1,
        'path': self.paths[index],
        'mapped_id': self.labels[index],
        'lens': waveform_length  # Add the waveform length to the sample
        }
        return sample

    def __len__(self):
        return len(self.paths)
    
    def collate_fn(self, batch):
        audios = [item['waveform'].squeeze(0) for item in batch]
        mapped_ids = [item['mapped_id'] for item in batch]
        mapped_ids = torch.tensor(mapped_ids)
        waveform_lengths = [item['lens'] for item in batch]  # Collect lengths
        waveform_lengths = torch.tensor(waveform_lengths)
        audio_paths = [item['path'] for item in batch]

        audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

        return {
            "waveform": audios_padded,
            "mapped_id": mapped_ids,
            "lens": waveform_lengths,  # Return the lengths as part of the collate function
            "path": audio_paths,
        }
    
    
class Evaluation_Dataset(Dataset):
    def __init__(self, paths, root):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = paths
        self.root = root
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])
        waveform  = load_audio(path, -1)
        sample = {
            'waveform': torch.FloatTensor(waveform),
            'path': path,
            'lens': waveform.shape  # Add the waveform length to the sample
        }
        
        return sample

    def collate_fn(self, batch):
        audios = [item['waveform'] for item in batch]
        waveform_lengths = [item['lens'] for item in batch]  # Collect lengths
        waveform_lengths = torch.tensor(waveform_lengths)
        audio_paths = [item['path'] for item in batch]

        audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

        return {
            "waveform": audios_padded,
            "lens": waveform_lengths,  # Return the lengths as part of the collate function
            "path": audio_paths
        }
