from torch import nn
from torchsummary import summary


import torch.nn as nn


class CNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1,
        #         padding=2
        #     ),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )

        # Third convolutional block (new)
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
        #     ),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )

        # # Fourth convolutional block (new)
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=128,
        #         out_channels=256,
        #         kernel_size=3,
        #         stride=1,
        #         padding=2
        #     ),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # After the conv layers, the feature map size is changed.
        # Assuming the input shape is [batch, 1, 13, 173]:
        # conv1: output -> [32, 15, 175] then maxpool -> [32, 7, 87]
        # conv2: output -> [64, 9, 89] then maxpool -> [64, 4, 44]
        # conv3: output -> [128, 6, 46] then maxpool -> [128, 3, 23]
        # conv4: output -> [256, 5, 25] then maxpool -> [256, 2, 12]
        # Thus, the flattened feature vector size is 256*2*12 = 6144.
        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 23, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.linear_2 = nn.Linear(64, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # print(x.shape)
        x = self.dense_layer(x)
        logits = self.linear_2(x)
        predictions = self.softmax(logits)
        return predictions


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(87*20, 512)
        # self.fc2 = nn.Linear(512, 256)

        self.fc4 = nn.Linear(512, 8)

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.batchnorm1(self.fc1(x)))
        # x = self.relu(self.batchnorm2(self.fc2(x)))
        # x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.fc4(x)
        return self.softmax(x)


if __name__ == "__main__":

    model = MLP()

    summary(model.cuda(), (1, 87, 20))
