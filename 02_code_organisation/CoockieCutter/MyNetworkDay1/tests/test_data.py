import torch
import numpy as np
import pytest
#import os
#print(os.listdir("../data/processed/training.pt"))
train_data, train_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/02_code_organisation/CoockieCutter/MyNetworkDay1/data/processed/training.pt')
train_data = torch.unsqueeze(train_data,1)
#trainset = torch.utils.data.TensorDataset(*trainset)
test_data, test_label = torch.load('/Users/frederikkjaer/Documents/DTU/8Semester/MLOps/dtu_mlops/02_code_organisation/CoockieCutter/MyNetworkDay1/data/processed/test.pt')
test_data = torch.unsqueeze(test_data,1)

#testset = torch.utils.data.TensorDataset(*testset)
@pytest.mark.parametrize("test_input,expected", [("train_data", torch.Size([1,28,28])), ("test_data", torch.Size([1,28,28]))])
def test_data_shape(test_input, expected):
    assert eval(test_input)[0].shape == expected


@pytest.mark.parametrize("test_input,expected", [("train_label", np.arange(10)), ("test_label", np.arange(10))])
def test_label_values(test_input, expected):
    assert np.array_equal(np.unique(eval(test_input)), expected)
