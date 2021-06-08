import torch
import torchvision
import wandb
from torchvision import datasets, transforms
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
wandb.init()

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# Writer will output to ./runs/ directory by default
loss_list = []
model = MyAwesomeModel()
wandb.watch(model, log_freq=100)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=float(0.003))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
epochs = 5
steps = 0
model.train()
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        grid = torchvision.utils.make_grid(images)
        wandb.log({"loss": loss})
    wandb.log({"Input Image" : [wandb.Image(i) for i in images]})
    

#from torch.utils.tensorboard import SummaryWriter
#import numpy as np

#writer = SummaryWriter()

#for n_iter in range(100):
#    writer.add_scalar('Loss/train', np.random.random(), n_iter)
#    writer.add_scalar('Loss/test', np.random.random(), n_iter)
#    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)