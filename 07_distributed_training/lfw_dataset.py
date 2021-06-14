"""
LFW dataloading
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform: None):
        folders_dir = [os.path.join(path_to_folder, o) for o in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder,o))]
        images_dir = []
        for i in range(len(folders_dir)):
            for f in os.listdir(folders_dir[i]):
                if os.path.isfile(os.path.join(folders_dir[i], f)):
                    images_dir.append(os.path.join(folders_dir[i], f)) #for f in os.listdir(folders_dir[i]) if os.path.isfile(os.path.join(folders_dir[i], f))])
        self.transform = transform
        self.images = images_dir
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        image_dir = self.images[index]
        img = Image.open(image_dir)
        
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw-deepfunneled', type=str)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.25, 0.75), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=500, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        image = next(iter(dataloader))
        for i in range(image.shape[0]):
            plt.imshow(image[i].permute(1, 2, 0)  )
            plt.show()

    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            print('hej')
            start = time.time()
            for batch in dataloader:
                # simulate that we do something with the batch
                time.sleep(0.001)
            end = time.time()
            
            res.append(end - start)
            
        res = np.array(res)
        print('Timing:', np.mean(res),'+-',np.std(res))
        