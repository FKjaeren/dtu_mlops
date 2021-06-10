import torch.nn.functional as F
import torch
from torch import nn, optim

kernel_size = 5
padding = 2
channel_sizes = [1,6,16]

class MyAwesomeModel(nn.Module):
    def __init__(self):
        #Der mangler convolution layer
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = channel_sizes[0], out_channels = channel_sizes[1], kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv2d(
            in_channels = channel_sizes[1], out_channels = channel_sizes[2], kernel_size = kernel_size, padding = padding)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor but instead it is: ', x.ndim)
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28] but had: ', x.shape )
        # Now with dropout
        x = self.dropout(F.max_pool2d(F.relu(self.conv1(x)),(2,2)))
        x = self.dropout(F.max_pool2d(F.relu(self.conv2(x)),(2,2)))

        x = torch.flatten(x,1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
