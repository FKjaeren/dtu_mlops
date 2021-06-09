import sys
import torch
import pytest
sys.path.insert(1,'/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/02_code_organisation/CoockieCutter/MyNetworkDay1/src/models')
from model import MyAwesomeModel
model = MyAwesomeModel()
batch_size = 64

train_data, train_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/02_code_organisation/CoockieCutter/MyNetworkDay1/data/processed/training.pt')
train_data_unsqueezed = torch.unsqueeze(train_data,1)
trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*(train_data_unsqueezed,train_label)), batch_size=batch_size, shuffle=True)

test_data, test_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/02_code_organisation/CoockieCutter/MyNetworkDay1/data/processed/test.pt')
test_data_unsqueezed = torch.unsqueeze(test_data,1)
testloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*(test_data_unsqueezed,test_label)), batch_size=batch_size, shuffle=True)
def test_model(trainloader = trainloader):
    for images, labels in trainloader:
        assert images.shape == torch.Size([batch_size,1,28,28])
        output = model(images.float())
        assert output.shape == torch.Size([batch_size, 10])
        break

def test_raises_warnings():
    model = MyAwesomeModel()
    batch_size = 64
    with pytest.raises(ValueError):
        test1 = torch.randn(1,28,28)
        output = model(test1)
    with pytest.raises(ValueError):
        test2 = torch.randn(batch_size,3,28,28)
        output = model(test2)