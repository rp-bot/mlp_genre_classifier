import torch
from torch import nn
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader
from _dataset import PatchBanksDataset
from _model import MLP
from tqdm import tqdm

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        predictions = model(inputs)

        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("----------------------------")

    print("Training is Done")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    annotations = "data/PatchBanks/annotations.csv"
    audio_dir = "data/PatchBanks"
    SAMPLE_RATE = 44_100
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 1e-3

    mel_spec = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=20,
        melkwargs={

            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 64,
        }
    )

    pbd = PatchBanksDataset(annotations, audio_dir,
                            mel_spec, SAMPLE_RATE, 3, device)

    train_data_loader = DataLoader(pbd, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)

    mlp_network = MLP().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_network.parameters(), lr=LR)

    train(mlp_network, train_data_loader,
          loss_fn, optimizer, device, epochs=EPOCHS)

    torch.save(mlp_network.state_dict(),
               "trained_models/feedforwardnet.pth")

    print("model saved")
