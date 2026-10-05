"""
PyTorch Dataset, Balanced Batch Sampler, and Color-Invariant Augmentations for Sarees.
"""

import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomChannelPermutation:
    """Randomly shuffles the color channels with probability p."""
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)
        return img


class SareePatternDataset(Dataset):
    """
    Dataset for Saree Designs.
    Expects data_dir with class folders, each containing multiple colorway images.
    """
    def __init__(self, root_dir: str, is_train: bool = True, img_size: int = 224):
        self.root_dir = root_dir
        self.is_train = is_train
        self.img_size = img_size

        # Find all design class folders
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.classes = class_dirs
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []  # list of (img_path, class_idx)
        self.class_indices = {i: [] for i in range(len(self.classes))}

        for cls_name in class_dirs:
            cls_idx = self.class_to_idx[cls_name]
            cls_path = os.path.join(root_dir, cls_name)
            img_files = sorted(glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png")))
            for img_p in img_files:
                sample_idx = len(self.samples)
                self.samples.append((img_p, cls_idx))
                self.class_indices[cls_idx].append(sample_idx)

        # Build transform pipelines
        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size + 16, img_size + 16)),
                T.RandomCrop((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                RandomChannelPermutation(p=0.3),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.4),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            tensor_img = self.transform(im)
        return tensor_img, label, img_path


class BalancedPKBatchSampler(Sampler):
    """
    Balanced P x K Batch Sampler for Metric Learning.
    In each batch, samples P distinct design classes, with K colorway instances per class.
    Batch size = P * K.
    """
    def __init__(self, class_indices: dict, p_classes: int = 8, k_instances: int = 4, iterations_per_epoch: int = 25):
        self.class_indices = class_indices
        self.classes = [c for c, idxs in class_indices.items() if len(idxs) >= 2]
        self.p = min(p_classes, len(self.classes))
        self.k = k_instances
        self.iterations = iterations_per_epoch

    def __iter__(self):
        for _ in range(self.iterations):
            selected_classes = random.sample(self.classes, self.p)
            batch = []
            for cls in selected_classes:
                indices = self.class_indices[cls]
                if len(indices) >= self.k:
                    batch.extend(random.sample(indices, self.k))
                else:
                    # Sample with replacement if fewer instances
                    batch.extend(random.choices(indices, k=self.k))
            yield batch

    def __len__(self) -> int:
        return self.iterations
