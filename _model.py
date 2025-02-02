from torch import nn
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*44, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.linear_2 = nn.Linear(128, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [1, 13, 173]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense_layer(x)
        logits = self.linear_2(x)
        predictions = self.softmax(logits)

        return predictions


if __name__ == '__main__':

    model = MLP()

    summary(model.cuda(), (1, 13, 173))
