import torch
import argparse
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.distributions import Uniform
import glob
from torchvision import datasets
from torch.utils.data import DataLoader 
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision.transforms import transforms
torch.manual_seed(100)
print("no errors")
import trainer1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_norm", default = True, type = bool)
    parser.add_argument("--batch_size_train", default = 60, type = int)
    parser.add_argument("--batch_size_val", default = 10, type = int)
    parser.add_argument("--criterion_type", default = "cross-entropy", type = str)
    parser.add_argument("--data_path", default = "./datasets/MNIST/", type = str)
    parser.add_argument("--download_data", default = False, type = bool)
    parser.add_argument("--init_weights", default = True, type = bool)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--num_classes", default = 10, type = int)
    parser.add_argument("--num_epochs", default = 100, type = int)
    parser.add_argument("--num_val_examples", default = 1000, type = int)
    parser.add_argument("--optimizer_type", default = "Adam", type = str)
    parser.add_argument("--progress", default = True, type = bool)
    parser.add_argument("--seed", default = 0, type = int)


    args = parser.parse_args()
    path_train=r'C:\Users\asus\OneDrive\Desktop\ADA\train_set'
    path_val=r'C:\Users\asus\OneDrive\Desktop\ADA\val_set'
    path_test=r'C:\Users\asus\OneDrive\Desktop\ADA\test_set'

  


    #defining transformations 1 and 2
    Transform=transforms.Compose([
    
    transforms.ToTensor()
    
    ])

    
    ##training set with download set to true
    train_set = datasets.MNIST(path_train, download=True, train=True, transform=Transform)
    
    #testing set with download set to true
    val_set = datasets.MNIST(path_val, download=True, train=False, transform=Transform)


    test_set=datasets.MNIST(path_test,download=True, train=False, transform=Transform)

    #train loader with batch size 32 which gives us batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True)
    
    #val loader with batch size 32 which gives us batches of 32
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_val, shuffle=True)
    print(len(train_loader))
    print(len(val_loader))

    mean=torch.tensor(0)
    std=torch.tensor(0)
    nz=6

    trainer=trainer1.trainer_class(train_loader,val_loader,args)
    model=trainer.train()
    

    
    

    






            


