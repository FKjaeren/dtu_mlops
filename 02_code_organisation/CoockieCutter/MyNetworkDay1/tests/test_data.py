from torchvision import datasets, transforms
import torch
#import os
#print(os.listdir("../data/processed/training.pt"))
train_data, train_label = torch.load('../data/processed/training.pt')
train_data = torch.unsqueeze(train_data,1)
#trainset = torch.utils.data.TensorDataset(*trainset)
test_data, test_label = torch.load('../data/processed/test.pt')
test_data = torch.unsqueeze(test_data,1)
#testset = torch.utils.data.TensorDataset(*testset)
assert len(train_data) == 60000 
assert len(test_data) == 10000

#assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
#assert that all labels are represented

