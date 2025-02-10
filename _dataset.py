from torch.utils.data import Dataset, Sampler, DataLoader, random_split
import pandas as pd
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram, Resample
import torch
import numpy as np
import matplotlib.pyplot as plt


class PatchBanksDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        resampler,
        target_sample_rate,
        duration,
        device,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.input_duration = duration
        self.device = device
        self.tranformation = transformation.to(self.device)
        self.resampler = resampler.to(self.device)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.annotations.iloc[index, 1]
        label = self.annotations.iloc[index, 3]
        signal, sr = torchaudio.load(audio_sample_path, normalize=True)
        signal = signal.to(self.device)

        # trim to only 5 seconds
        num_samples = self.input_duration * self.target_sample_rate
        signal = signal[:, :num_samples]

        fade_samples = int(0.25 * self.target_sample_rate)
        fade_curve = torch.linspace(1, 0, fade_samples).to(signal.device)

        signal[:, -fade_samples:] *= fade_curve

        # Ensure signal is stereo
        signal = self.resampler(signal)
        signal = self.tranformation(signal)

        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        signal = signal.transpose(2, 1)

        # signal = torch.log1p(signal)
        return signal, label


class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=42):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = len(np.unique(self.labels))

        # Create a dictionary mapping each class to its indices
        self.class_indices = {
            label: np.where(self.labels == label)[0].tolist()
            for label in np.unique(self.labels)
        }

        # Set seed for reproducibility
        np.random.seed(self.seed)

        # Shuffle indices within each class
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])

    def __iter__(self):
        batch = []

        while any(
            len(indices) > 0 for indices in self.class_indices.values()
        ):  # Stop when all samples are used
            for label, indices in self.class_indices.items():
                if indices:  # Ensure there are samples left
                    batch.append(indices.pop())

                if len(batch) == self.batch_size:  # If batch is full, yield it
                    yield batch
                    batch = []

        # Yield any remaining samples (if batch size isn't a perfect fit)
        if batch:
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size


if __name__ == "__main__":

    annotations = "data/PatchBanks/annotations.csv"
    audio_dir = "data/PatchBanks"
    SAMPLE_RATE = 44_100
    TARGET_SAMPLE_RATE = 22_050
    batch_size = 128

    # mel_spec = MFCC(
    #     sample_rate=SAMPLE_RATE,
    #     n_mfcc=20,
    #     melkwargs={
    #         "n_fft": 1024,
    #         "hop_length": 512,
    #         "n_mels": 40,
    #     },
    # )

    resampler = Resample(SAMPLE_RATE, TARGET_SAMPLE_RATE)

    mel_spec = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbd = PatchBanksDataset(
        annotations, audio_dir, mel_spec, resampler, SAMPLE_RATE, 2, device
    )

    train_size = int(0.7 * len(pbd))
    test_size = len(pbd) - train_size
    train_dataset, test_dataset = random_split(pbd, [train_size, test_size])

    train_labels = [pbd.annotations.iloc[i, 3] for i in train_dataset.indices]
    test_labels = [pbd.annotations.iloc[i, 3] for i in test_dataset.indices]

    train_sampler = StratifiedBatchSampler(train_labels, batch_size, seed=42)
    test_sampler = StratifiedBatchSampler(test_labels, batch_size, seed=42)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    print(f"Train set size: {train_size}, Test set size: {test_size}")

    # How many train and test samnpels for each class

    # plt.figure(figsize=(10, 4))
    # plt.imshow(pbd[0][0][0].cpu().detach().numpy(), cmap="viridis", aspect="auto")
    # # plt.title(f'MFCC {class_label}')
    # plt.ylabel("Mel Bands")
    # plt.xlabel("Time")
    # plt.colorbar(format="%+2.0f dB")
    # # plt.gca().invert_yaxis()
    # plt.savefig(f"plots/features/MFCC_feature_dsag.png")

    # for i, class_label in [(0, "house"), (1101, "tr_808"), (2201, "tr_909"), (3301, "hiphop"), (4401, "pop rock"), (5501, "retro"), (6601, "latin percussions"), (7701, "samba")]:
    #     plt.figure(figsize=(10, 4))
    #     plt.imshow(pbd[i][0][0].cpu().detach().numpy(),
    #                cmap='viridis', aspect='auto')
    #     plt.title(f'MFCC {class_label}')
    #     plt.ylabel('Cepstrum Coefficients')
    #     plt.xlabel('Time')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.gca().invert_yaxis()
    #     plt.savefig(f"plots/features/MFCC_feature_{class_label}.png")

    # plt.plot(pbd[0][0][0].cpu().detach().numpy())
    # plt.savefig("example.png")
