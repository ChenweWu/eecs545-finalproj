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
        is_flag = 0
        for j in range(5):
            if tensor[i,j] == 1:
                labels[i] = j
                is_flag = 1
                break
        if is_flag == 0:
            labels[i] = 5
        
    return labels.to(tensor.device)
def tensor_to_demo_labels(tensor):

    labels = tensor[:,7].long().to(tensor.device)
    return labels
def cli_main():
    seed_everything(42)


    transform = DataTransforms(True, 512)
    train_dataset = MultimodalPretrainingDataset_train_disc(split = "train", transform=transform)
    test_dataset = MultimodalPretrainingDataset_train_disc(split = "valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn = multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn = multimodal_collate_fn)
    
    import torchxrayvision as xrv
    
    model = models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(2048, 6)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    
    def train_model(model, criterion, optimizer, num_epochs=10):
        print("start to train the model")
        device = "cuda"
        model.to(device).train() 
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images, meta_data = batch['image'].to(device), batch['meta_data'].to(device)
                labels = tensor_to_labels(meta_data)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                print("hahahahahah, loss:",loss.item(), outputs.shape)
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                print(predicted, labels)
                correct = (predicted == labels).sum().item()
                print("acc:", correct*1.0/labels.shape[0])
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), 'model.pth')
    def evaluate_model(model):
        device = "cuda"
        model.eval().to(device) 
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                images, meta_data = batch['image'].to(device), batch['meta_data'].to(device)
                labels = tensor_to_labels(meta_data)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    train_model(model, criterion, optimizer, num_epochs=15)
    evaluate_model(model)
    


if __name__ == "__main__":
    cli_main()
