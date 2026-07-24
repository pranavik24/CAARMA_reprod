import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import Evaluation_Dataset, Train_Dataset
from .augmentation import Augmentation
from .voxceleb_split import load_validation_trials


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _effective_train_drop_last(config, dataset_length):
    drop_last = _as_bool(config.get('train_drop_last', True))
    if drop_last:
        return True

    batch_size = int(config['batch_size'])
    if batch_size > 1 and dataset_length % batch_size == 1:
        print(
            "Enabling train drop_last because the final training batch would "
            "contain one sample, which breaks BatchNorm."
        )
        return True

    return False


class super_dataset(LightningDataModule):
    def __init__(
        self,
        config,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.config = config


    def train_dataloader(self) -> DataLoader:
        # augmentation = Augmentation(add_noise=self.config['augmentations']['add_noise'], add_reverb=self.config['augmentations']['add_reverb'], drop_freq=self.config['augmentations']['drop_freq'], drop_chunk=self.config['augmentations']['drop_chunk'])
        train_dataset = Train_Dataset(self.config, self.config['second'], do_augmentation=self.config['do_augmentation'], augmentation=None) #augmentation)
        drop_last = _effective_train_drop_last(self.config, len(train_dataset))
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.config['num_workers'],
                batch_size=self.config['batch_size'],
                pin_memory=True,
                drop_last=drop_last,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        trials, root = load_validation_trials(self.config)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        # eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        eval_dataset = Evaluation_Dataset(eval_path, root=root)
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=int(self.config.get("val_num_workers", min(int(self.config["num_workers"]), 4))),
                                             shuffle=False, 
                                             batch_size=1)
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
