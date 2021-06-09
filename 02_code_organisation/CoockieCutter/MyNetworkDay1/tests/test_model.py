import sys
import torch
sys.path.insert(1,'../src/models/')
from model import MyAwesomeModel
model = MyAwesomeModel()
batch_size = 64

train_data, train_label = torch.load('../data/processed/training.pt')
train_data_unsqueezed = torch.unsqueeze(train_data,1)
trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*(train_data_unsqueezed,train_label)), batch_size=batch_size, shuffle=True)

test_data, test_label = torch.load('../data/processed/test.pt')
test_data_unsqueezed = torch.unsqueeze(test_data,1)
testloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*(test_data_unsqueezed,test_label)), batch_size=batch_size, shuffle=True)
def test_model(trainloader = trainloader):
    for images, labels in trainloader:
        assert images.shape == torch.Size([batch_size,1,28,28])
        output = model(images.float())
        assert output.shape == torch.Size([batch_size, 10])
        break

test_model()