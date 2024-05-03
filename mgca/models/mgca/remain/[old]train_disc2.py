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
from mgca.models.ssl_finetuner import SSLFineTuner_demo, SSLFineTuner_age, SSLFineTuner_disease
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset2,
                                            multimodal_collate_fn)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assuming inputs are raw logits of shape (batch_size, num_classes)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the exponential of the negative cross entropy loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# Example of using FocalLoss
# loss_fn = 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def tensor_to_labels(tensor):

    labels = torch.zeros(tensor.size(0), dtype=torch.long)

    max_indices = torch.argmax(tensor[:, :6], dim=1)

    labels = torch.where(torch.sum(tensor, dim=1) == 0, 6, max_indices)
    return labels.to(tensor.device)
class ResNet200D(nn.Module):
    def __init__(self, n_classes, model_name='resnet200d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output
def cli_main():
    seed_everything(42)

    print("chenwei test")
    transform = DataTransforms(True, 512)
    train_dataset = MultimodalPretrainingDataset2(split = "train", transform=transform)
    test_dataset = MultimodalPretrainingDataset2(split = "valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn = multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn = multimodal_collate_fn)
    
    import torchxrayvision as xrv
    # model = xrv.models.ResNet(weights="resnet50-res512-all")
    
    model = ResNet200D(n_classes=6)

    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    
    def train_model(model, criterion, optimizer, num_epochs=10):
        print("start to train the model")
        device = "cuda"
        model.to(device).train() 
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images, meta_data = batch['image'].to(device), batch['meta_data'].to(device)
                # print(meta_data)
                labels = tensor_to_labels(meta_data)
                print(labels)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                print("hahahahahah, loss:",loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                print("acc:", correct*1.0/labels.shape[0])
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
    train_model(model, criterion, optimizer, num_epochs=10)

    


if __name__ == "__main__":
    cli_main()
