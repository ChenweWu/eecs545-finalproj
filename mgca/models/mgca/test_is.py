import torch
import numpy as np
import torchxrayvision as xrv
from torchvision.models import inception_v3, resnet50
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from scipy.stats import entropy
from mgca.datasets.syn_pretrain_dataset import (SynPretrainingDataset,
                                           syn_multimodal_collate_fn)
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
import torchvision.transforms as transforms
def calculate_is(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        count = 0
        for batch in loader:
            images = batch['image'].to(device)
            outputs = model(images)
            preds.append(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
            count +=1
            if count > 100:
                break
    preds = np.concatenate(preds, axis=0)

    py = np.mean(preds, axis=0)

    scores = []
    for i in range(preds.shape[0]):
        scores.append(entropy(preds[i, :], py))

    is_score = np.exp(np.mean(scores))
    return is_score
if __name__ == "__main__":
    
    model = xrv.models.ResNet(weights="resnet50-res512-all") # if use this, remember to change model(x) to model.features(x)
    
    fake_dataset = SynPretrainingDataset(transform=None, data_pct=1.0,
                     imsize=512, max_words=77, is_single = 1)
    fake_dataset = SynPretrainingDataset(transform=DataTransforms(0,512), data_pct=1.0,
                     imsize=512, max_words=77, is_single = 0) # for inception-v3
    fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last = True, collate_fn = syn_multimodal_collate_fn)


    model = inception_v3(pretrained=True,transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    device = "cuda"
    model.to(device)
    
    IS_score = calculate_is(model, fake_loader, device)
    print('IS score:', IS_score)
