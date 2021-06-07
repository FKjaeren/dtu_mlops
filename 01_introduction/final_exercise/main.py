import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import mnist
from model import MyAwesomeModel
from torch import nn, optim
import torch.nn.functional as F

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        loss_list = []
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.003)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
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
            else:
                loss_list.append(running_loss/len(trainloader))
                print(f"Training loss: {running_loss/len(trainloader)}")
        plt.figure()
        epoch = np.arange(len(loss_list))
        print(len(loss_list))
        print(epoch)
        plt.plot(epoch, loss_list)
        plt.legend(['Training loss'])
        plt.xlabel('Epochs'), plt.ylabel('Loss')
        plt.show()
        torch.save(model, 'model.pth')

    def evaluate(self):
        accuracy_list = []
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        model.eval()
        with torch.no_grad():
            for images, labels in testloader:
                #images, labels = next(iter(testloader))
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                accuracy_list.append(accuracy.item()*100)
            else:
                print(f'Accuracy: {accuracy.item()*100}%')
        epoch = np.arange(len(accuracy_list))
        print("mean of accuracy = ", np.mean(accuracy_list))
        plt.figure()
        plt.plot(epoch, accuracy_list)
        plt.legend(['Test set accuacy'])
        plt.xlabel('Epochs'), plt.ylabel('Accuacy')
        plt.show()
        torch.save(model, 'model.pth')
if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    