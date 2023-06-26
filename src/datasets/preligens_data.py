import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision

from src.datasets.instance_dataset import InstanceDataset

_GLOBAL_SEED = 0
logger = getLogger()


def make_preligens_data(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
):
    dataset = InstanceDataset(
        root=root_path,
        transforms=transform,
        subset=("train" if training else "val"))
    logger.info('Preligens dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Preligens unsupervised data loader created')
    return dataset, data_loader, dist_sampler