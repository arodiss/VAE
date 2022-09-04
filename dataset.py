from typing import List, Optional, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset as BaseDataset
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from random import choice


class Dataset(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = OlivettiDataset(
            start=0,
            end=400,
            maybe_flip=True,
            maybe_saturate=0.1,
        )

        self.val_dataset = OlivettiDataset(
            start=0,
            end=400,
            maybe_flip=True,
            maybe_saturate=0.1,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class OlivettiDataset(BaseDataset):
    def __init__(self, start, end, maybe_flip, maybe_saturate=None):
        self.maybe_flip = maybe_flip
        self.maybe_saturate = maybe_saturate
        self.faces = fetch_olivetti_faces()['images'][start:end]
        np.random.shuffle(self.faces)

    def __len__(self):
        return self.faces.shape[0]

    def __getitem__(self, item):
        img = self.faces[item]
        if self.maybe_flip and choice([True, False]):
            img = np.fliplr(img)
            img = np.copy(img)
        if self.maybe_saturate is not None:
            img = img + np.random.normal(0, self.maybe_saturate, 1).astype(np.float32)
        return np.expand_dims(img, axis=0)
