import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224, split="train"):
        super().__init__()
        self.split = split
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split=self.split, transform=transform, data_pct=self.data_pct)
        
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct)
        
    
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=1)
        print(dataset[0])
        '''
        dataset = self.dataset(
            split="train", transform=transform, data_pct=0.01)
            # for sample a part of training data for testing
        from torch.utils.data import Subset
        from pytorch_lightning import seed_everything
        import random
        seed_everything(42)
        total_size = len(dataset)
        subset_size = int(0.01 * total_size)
        indices = list(range(total_size))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        dataset = Subset(dataset, subset_indices)
        '''
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )