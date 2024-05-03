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
from mgca.datasets.syn_pretrain_dataset import (SynPretrainingDataset,
                                           syn_multimodal_collate_fn)
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from tqdm import tqdm 

def cli_main():
    

    transform = DataTransforms(False, 512)
    test_dataset = SynPretrainingDataset(split = "valid", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn = multimodal_collate_fn)

    model = models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(torch.load('./data/disc/disease/disc_d1.pth')) # choose the classifier
    
    
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

    evaluate_model(model)
    

if __name__ == "__main__":
    cli_main()
