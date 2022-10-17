import os
from collections import Counter

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/dafre/",
        size: int = 224,
        min_scale: float = 0.8,
        batch_size: int = 32,
        workers: int = 4,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        erase_prob: float = 0.0,
        use_trivial_aug: bool = False,
        use_balanced_sampler: bool = False,
    ):
        """Classification Datamodule

        Args:
            root: Path to DAF:Re dataset
            size: Image size
            min_scale: Min crop scale
            batch_size: Number of batch samples
            workers: Number of data loader workers
            rand_aug_n: RandAugment number of augmentations
            rand_aug_m: RandAugment magnitude of augmentations
            erase_prob: Probability of applying random erasing
            use_trivial_aug: Apply TrivialAugment instead of RandAugment
            use_balanced_sampler: Use a class balanced data sampler
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.size = size
        self.min_scale = min_scale
        self.batch_size = batch_size
        self.workers = workers
        self.rand_aug_n = rand_aug_n
        self.rand_aug_m = rand_aug_m
        self.erase_prob = erase_prob
        self.use_trivial_aug = use_trivial_aug
        self.use_balanced_sampler = use_balanced_sampler

        # Define augmentations
        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (self.size, self.size), scale=(self.min_scale, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide()
                if self.use_trivial_aug
                else transforms.RandAugment(self.rand_aug_n, self.rand_aug_m),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=self.erase_prob),
            ]
        )
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = ImageFolder(
                root=os.path.join(self.root, "train"), transform=self.transforms_train
            )
            self.val_dataset = ImageFolder(
                root=os.path.join(self.root, "val"), transform=self.transforms_test
            )
        elif stage == "test":
            self.test_dataset = ImageFolder(
                root=os.path.join(self.root, "test"), transform=self.transforms_test
            )

    def train_dataloader(self):
        if self.use_balanced_sampler:
            sampler = self.get_balanced_sampler()
        else:
            sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.use_balanced_sampler,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def get_balanced_sampler(self):
        """Returns a balanced data sampler for the training dataset"""
        # Calculate class weights proportional to the number of samples per class
        targets = self.train_dataset.targets
        samples_per_class = list(Counter(targets).values())
        class_weights = 1 / torch.tensor(samples_per_class, dtype=torch.float)
        class_weights_targets = class_weights[targets]

        return WeightedRandomSampler(
            weights=class_weights_targets,  # type:ignore
            num_samples=len(class_weights_targets),
            replacement=True,
        )


if __name__ == "__main__":
    # dm = DataModule(root="data/dafre")
    dm = DataModule(
        root="data/dafre",
        use_balanced_sampler=True,
    )
    dm.setup()
    dl = dm.train_dataloader()
    # print(len(dm.train_dataset))
    # print(len(dm.val_dataset))

    for x, y in dl:
        # print(x.size())
        # print(x.min())
        # print(x.max())
        # print(x.dtype)
        print(y)
        print(y.size())
        print(y.dtype)
        break
