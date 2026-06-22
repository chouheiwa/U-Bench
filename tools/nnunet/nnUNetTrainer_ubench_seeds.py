"""Seeded nnU-Net trainers for the U-Bench comparison.

Each subclasses an official epoch-budget preset and only seeds the RNGs in
__init__ (mirrors main.py:58-68: random / numpy / torch / cuda). We deliberately
do NOT force cudnn.deterministic — that stays nnU-Net-native (benchmark on), so
the only thing the seed changes is run-to-run randomness, keeping nnU-Net's own
speed/behaviour intact. Installed into nnunetv2's variants dir so the framework
discovers the class by name (recursive_find_python_class).

Naming: nnUNetTrainer_s<seed>_<budget> e.g. nnUNetTrainer_s41_250e.
The _5e variants exist only for the smoke test.
"""
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
    nnUNetTrainer_5epochs,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class nnUNetTrainer_s41_250e(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        _seed_everything(41)
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_s42_250e(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        _seed_everything(42)
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_s43_250e(nnUNetTrainer_250epochs):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        _seed_everything(43)
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_s41_5e(nnUNetTrainer_5epochs):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        _seed_everything(41)
        super().__init__(plans, configuration, fold, dataset_json, device)
