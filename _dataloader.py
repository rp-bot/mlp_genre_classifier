import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class PatchBanksDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 1]
        label = self.annotations.iloc[index, 3]
        signal, sr = torchaudio.load(audio_sample_path)

        return signal, label


if __name__ == '__main__':

    annotations = "data/PatchBanks/annotations.csv"
    audio_dir = "data/PatchBanks"

    pbd = PatchBanksDataset(annotations, audio_dir)

    print(f"THere are {len(pbd)} samples")

    signal, label = pbd[0]

    print(signal.shape)
