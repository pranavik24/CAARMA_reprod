import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import Evaluation_Dataset, Train_Dataset
from .augmentation import Augmentation


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.config['num_workers'],
                batch_size=self.config['batch_size'],
                pin_memory=True,
                drop_last=_as_bool(self.config.get('train_drop_last', True)),
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        trials = np.loadtxt(self.config['trial_path'], str)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        # eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        eval_dataset = Evaluation_Dataset(eval_path, root=self.config['root'])
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=10,
                                             shuffle=False, 
                                             batch_size=1)
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
