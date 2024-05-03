import datetime
import os
from argparse import ArgumentParser

import torch
import torchvision.models as models
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.mgca.mgca_module import MGCA

from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset_train_disc,
                                            multimodal_collate_fn)
from mgca.datasets.syn_pretrain_dataset import (SynPretrainingDataset,
                                           syn_multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
        
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def tensor_to_labels(tensor):

    labels = torch.zeros(tensor.size(0), dtype=torch.long)

    for i in range(tensor.shape[0]):
        if tensor[i,0] == 1:
            labels[i] = 1
    return labels.to(tensor.device)
def tensor_to_demo_labels(tensor):

    labels = tensor[:,7].long().to(tensor.device)
    return labels
def cli_main():
    seed_everything(42)


    transform = DataTransforms(False, 512)
    train_dataset = SynPretrainingDataset(split = "train", transform=transform)
    test_dataset = SynPretrainingDataset(split = "train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn = syn_multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn = syn_multimodal_collate_fn)
    
    import torchxrayvision as xrv
    #model = xrv.models.ResNet(weights="resnet50-res512-all")
    
    model = models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(2048, 2)
    d_dict = torch.load('./data/disc/disease/disc_d1.pth')
    model.load_state_dict(d_dict)

    
    
    def evaluate_model(model):
        device = "cuda"
        model.eval().to(device) 
        correct = 0
        total = 0
        total_pos = 0
        correct_pos = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                images, meta_data = batch['image'].to(device), batch['meta_data'].to(device)
                labels = tensor_to_labels(meta_data)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(labels, predicted)
                for idx in range(labels.shape[0]):
                    if labels[idx] == 1:
                        total_pos +=1
                        if predicted[idx] == 1:
                            correct_pos +=1
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
        print(f'TPR of the network on the 10000 test images: {100 * correct_pos / total_pos:.2f}%')
    
    evaluate_model(model)
    


if __name__ == "__main__":
    cli_main()
