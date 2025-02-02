import torch
from torch import nn
from torchaudio.transforms import MFCC
from torch.utils.data import DataLoader, random_split
from _dataset import PatchBanksDataset
from _model import MLP
from tqdm import tqdm

class_mappings = [
    "edm",
    "tr-808",
    "tr-909",
    "hip-hop",
    "pop rock",
    "retro",
    "latin percussions",
    "samba",
]


def evaluate(model, test_data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_data_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            predicted = torch.argmax(predictions, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(test_data_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 1e-10
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(predictions, dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
    print("----------------------------")


def train(model, train_data_loader, loss_fn, optimizer, device, epochs):
    model.train()
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, train_data_loader, loss_fn, optimizer, device)
        print("----------------------------")
    print("Training is Done")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    annotations = "data/PatchBanks/annotations.csv"
    audio_dir = "data/PatchBanks"
    SAMPLE_RATE = 44_100
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 1e-3

    mfcc_transform = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=13,
        melkwargs={
            "n_fft": 1024,
            "hop_length": 512,
            "n_mels": 40,
        },
    )

    pbd = PatchBanksDataset(
        annotations, audio_dir, mfcc_transform, SAMPLE_RATE, 2, device
    )

    dataset = PatchBanksDataset(
        annotations, audio_dir, mfcc_transform, SAMPLE_RATE, 2, device
    )
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mlp_network = MLP().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_network.parameters(), lr=LR)

    print("Training Model")
    train(mlp_network, train_data_loader, loss_fn, optimizer, device, epochs=EPOCHS)
    print("============================")

    print("Final Evaluation on Test Set:")
    evaluate(mlp_network, test_data_loader, loss_fn, device)
    print("============================")

    print("saving Model")
    torch.save(mlp_network.state_dict(), "trained_models/MLP.pth")
    print("model saved")
