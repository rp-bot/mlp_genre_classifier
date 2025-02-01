import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        return signal, label

    def _get_audio_sample_path(self, index):
        audio_file_path = self.annotations.iloc[index, 0]
        return audio_file_path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == '__main__':

    a = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    audi_dir = "data/UrbanSound8K/audio"

    usd = UrbanSoundDataset(a, audi_dir)

    print(f"THere are {len(usd)} samples")

    signal, label = usd[0]

    print(signal)
