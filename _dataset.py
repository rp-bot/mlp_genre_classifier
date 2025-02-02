import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from torchaudio.transforms import MFCC
import torch
import matplotlib.pyplot as plt


class PatchBanksDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, duration, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.input_duration = duration
        self.device = device
        self.tranformation = transformation.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 1]
        label = self.annotations.iloc[index, 3]
        signal, sr = torchaudio.load(audio_sample_path, )
        signal = signal.to(self.device)

        # trim to only 5 seconds
        num_samples = self.input_duration * self.target_sample_rate
        signal = signal[:, :num_samples]

        fade_samples = int(0.25 * self.target_sample_rate)
        fade_curve = torch.linspace(1, 0, fade_samples).to(signal.device)

        signal[:, -fade_samples:] *= fade_curve

        # Ensure signal is stereo

        signal = self.tranformation(signal)

        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0 ,keepdim=True)

        return signal, label


if __name__ == '__main__':

    annotations = "data/PatchBanks/annotations.csv"
    audio_dir = "data/PatchBanks"
    SAMPLE_RATE = 44_100

    mel_spec = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=20,
        melkwargs={

            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 64,
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbd = PatchBanksDataset(annotations, audio_dir,
                            mel_spec, SAMPLE_RATE, 3, device)

    print(pbd[0][0].shape)
