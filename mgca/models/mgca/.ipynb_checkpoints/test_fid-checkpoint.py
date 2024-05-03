import torch
import numpy as np
import torchxrayvision as xrv
from torchvision.models import inception_v3, resnet50
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from mgca.datasets.syn_pretrain_dataset import (SynPretrainingDataset,
                                           syn_multimodal_collate_fn)
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms

import torchvision.transforms as transforms
def calculate_fid(model, real_loader, fake_loader, device):
    model.eval()
    real_features = []
    fake_features = []
    count = 0
    with torch.no_grad():
        for real_images, fake_images in zip(real_loader, fake_loader):
            real_images = real_images['image'].to(device)
            fake_images = fake_images['image'].to(device)
            real_batch_features = model.features(real_images)
            fake_batch_features = model.features(fake_images)
            real_features.append(real_batch_features.cpu().numpy())
            fake_features.append(fake_batch_features.cpu().numpy())
            count = count + 1
            print("count:", count)
            if count*32 >= 200:
                break
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    print(real_features.shape)
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    print(mu_real, mu_fake)
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    
    return fid
if __name__ == "__main__":
    
    model = xrv.models.ResNet(weights="resnet50-res512-all") # if use this, remember to change model(x) to model.features(x)
    

    real_dataset = MultimodalPretrainingDataset(transform=None, data_pct=1.0,
                     imsize=512, max_words=77, is_single = 1)
    fake_dataset = SynPretrainingDataset(transform=None, data_pct=1.0,
                     imsize=512, max_words=77, is_single = 1)
    
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last = True, collate_fn = multimodal_collate_fn)
    fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last = True, collate_fn = syn_multimodal_collate_fn)

    
    #model = resnet50(pretrained=True)
    #model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    device = "cuda"
    model.to(device)
    
    fid_score = calculate_fid(model, real_loader, fake_loader, device)
    print('FID score:', fid_score)
