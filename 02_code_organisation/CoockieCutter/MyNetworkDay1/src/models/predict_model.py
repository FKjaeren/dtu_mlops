import torch
test_set = torch.load('../../data/processed/test.pt')
testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_set), batch_size=1, shuffle=True)
def predict_class(data = testloader):
    model = torch.load('../../models/model.pth')
    model.eval()
    test_picture, test_label = next(iter(data))
    print("test_picture shape", test_picture.shape)
    prediction = model(test_picture.float())
    ps = torch.exp(prediction)
    top_p, top_class = ps.topk(1, dim=1)
    print(top_class)
predict_class()
