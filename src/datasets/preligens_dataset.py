"""Implementation of the parent class for preligens Datasets."""
from abc import ABC
from pathlib import Path

from torch.utils.data import Dataset


class PreligensDataset(ABC, Dataset):
    """Abstract parent class to manage Preligens datasets (classification, instance, segmentation...)."""

    def __init__(self, root_dir: Path):
        pass
