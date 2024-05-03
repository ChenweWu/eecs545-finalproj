import datetime
import os
from argparse import ArgumentParser
import torch
import pandas as pd
import json
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
        
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--data_pct", type=float, default=0.01)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 5

    seed_everything(args.seed)

    import torchxrayvision as xrv
    import skimage, torch, torchvision
    from torch.utils.data import DataLoader
    file_path = "/scratch/bcde/ztshuai/syn_roentgen/filenames.json" ################ hhhhhhh woc this is the path, don't remember to modify
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    filenames = pd.DataFrame(data)
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval().to("cuda")
    # Prepare the image:
    woc = {'Edema':0, 'Effusion':0, 'Consolidation':0, 'Pneumothorax':0, 'Pneumonia':0}
    sum_count = 0
    with torch.no_grad():
        for i in range(len(filenames)):
            sum_count +=1
            img = skimage.io.imread(filenames.loc[i]['path'])
            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
            img = img.mean(2)[None, ...] # Make single color channel

            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

            img = transform(img)
            img = torch.from_numpy(img).to("cuda")

            # Load model and process image

            outputs = model(img[None,...]) # or model.features(img[None,...]) 

            # Print results
            temp_dict = dict(zip(model.pathologies,outputs[0].detach().to("cpu").numpy()))
            keyyy = 'Edema'
            print(filenames.loc[i]['pa'])
            
            if (temp_dict[keyyy]>0.5)-filenames.loc[i]['pa'][0]==0:
                woc[keyyy] +=1
                if temp_dict[keyyy]>0.5:
                    print("Edema")
            keyyy = 'Effusion'
            if (temp_dict[keyyy]>0.5)-filenames.loc[i]['pa'][1]==0:
                woc[keyyy] +=1
                if temp_dict[keyyy]>0.5:
                    print("Effusion")
            keyyy = 'Consolidation'
            if (temp_dict[keyyy]>0.5)-filenames.loc[i]['pa'][2]==0:
                woc[keyyy] +=1
                if temp_dict[keyyy]>0.5:
                    print("Consolidation")
            keyyy = 'Pneumothorax'
            if (temp_dict[keyyy]>0.5)-filenames.loc[i]['pa'][3]==0:
                woc[keyyy] +=1
                if temp_dict[keyyy]>0.5:
                    print("Pneumothorax")
            keyyy = 'Pneumonia'
            if (temp_dict[keyyy]>0.5)-filenames.loc[i]['pa'][4]==0:
                woc[keyyy] +=1
                if temp_dict[keyyy]>0.5:
                    print("Pneumonia")
            if sum_count % 100 == 0:
                print(sum_count)
            if sum_count > 50:
                break
                
    print(woc['Edema']*1.0/sum_count)
    print(woc['Effusion']*1.0/sum_count)
    print(woc['Consolidation']*1.0/sum_count)
    print(woc['Pneumothorax']*1.0/sum_count)
    print(woc['Pneumonia']*1.0/sum_count)

if __name__ == "__main__":
    cli_main()
