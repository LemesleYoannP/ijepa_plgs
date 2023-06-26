"""Implementation of the loading of an instance dataset from disk."""
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes
from tqdm.auto import tqdm

from torchvision.transforms.v2 import functional as F
from torchvision import datapoints


from src.datasets.preligens_dataset import PreligensDataset


class InstanceDataset(PreligensDataset):
    """Instance Dataset for PyTorch pre-training.

    Load information regarding dataset (images path, bounding boxes and labels) and store it in memory

    Attributes:
        root_dir: location of dataset root folder on drive (informative)
        img_size: resolution of the image - expected square (informative)
        img_metas : localisation of images (absolute path) and associated bboxes, labels
        labels_to_int : label keys and its integer correspondence
        int_to_labels : integer keys and its string correspondence
    """

    root_dir: Path = None
    img_size: float = 256.0
    img_metas: Dict[Path, np.ndarray] = {}
    labels_to_int: Dict[int, str] = {}
    int_to_labels: Dict[int, str] = {}

    def __init__(self, root_dir: str, img_size: int = 256, subset: str = "train", transforms = None):
        """Load a dataset from a root dir on disk.

        Data will be stored in memory as a dict with :
        - key : the location of the tile on disk
        - values : a ndarray containing bboxes coordinates and labels

        Args:
            root_dir: location on the dataset on disc
            img_size: size of the images (expected height equals width)
            subset : indicates if we want to load train, val or test data
        """
        root_dir = Path(root_dir)
        super().__init__(root_dir)

        self.root_dir = root_dir
        self.img_size = img_size

        with open(self.root_dir / "dataset.yaml", "r") as stream:
            dataset_yaml = yaml.safe_load(stream)

        self.int_to_labels = {k: v for k, v in enumerate(dataset_yaml["label_name"])}
        self.labels_to_int = {v: k for k, v in enumerate(dataset_yaml["label_name"])}

        pd_bounding_boxes = pd.read_csv(self.root_dir / subset / "bounding_boxes.csv")
        pd_bounding_boxes.columns = [
            "tile_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "pos",
            "target_height",
            "target_width",
            "angle",
            "height",
            "width",
            "x_cent",
            "y_cent",
        ]

        for tile_path, tile_info in tqdm(pd_bounding_boxes.groupby("tile_path"), desc=f"parsing {subset} data"):

            labels = tile_info[["xmin", "ymin", "xmax", "ymax", "label"]]

            if labels.isnull().all().all():  # no bounding box
                labels = None
            else:
                labels = labels.values
            self.img_metas[root_dir / tile_path] = labels

        self.img_keys = list(self.img_metas.keys())

        # implement later transform
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the len of the dataset.

        Returns:
            the len of the dataset (number of tiles)
        """
        return len(self.img_keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return dataset item of index idx.

        Args:
            idx: the index of the item

        Returns:
            the image, bboxes and labels
        """
        img_path = self.img_keys[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = (ToTensor()(image) * 256.0).to(torch.uint8)

        return image
