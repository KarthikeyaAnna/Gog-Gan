"""
Dataset classes and utilities for loading images (AFHQ Dog or local datasets).
"""

import os
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Dog breed caption templates for diverse text conditioning



def breed_from_folder_name(folder_name: str) -> str:
    """Convert folder name like 'n02085620-Chihuahua' to 'Chihuahua'."""
    parts = folder_name.split("-", 1)
    if len(parts) > 1:
        breed = parts[1].replace("_", " ")
    else:
        breed = folder_name.replace("_", " ")
    return breed


class AFHQDogDataset(Dataset):
    """
    AFHQ-Dog dataset for high-quality face-only generation.
    Filters the huggan/afhq dataset for label==1 (dogs).
    """

    def __init__(self, root_dir=None, image_size=64, split="train"):
        super().__init__()
        self.image_size = image_size
        self.is_eval = (split != "train")
        
        print(f"[Dataset] Loading AFHQ (Animal Faces High Quality) dataset from HuggingFace (split={split})...")
        from datasets import load_dataset
        # Load the dataset and filter only for dogs (label == 1)
        # AFHQ labels: 0=cat, 1=dog, 2=wild
        full_dataset = load_dataset("huggan/afhq", split=split)
        self.hf_dataset = full_dataset.filter(lambda x: x["label"] == 1)
        print(f"[Dataset] Filtered down to {len(self.hf_dataset)} dog faces.")

        # Image transforms
        self.transform = T.Compose([
            T.Resize(image_size + 8),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] range
        ])

        self.eval_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        transform = self.eval_transform if self.is_eval else self.transform

        sample = self.hf_dataset[idx]
        try:
            img = sample["image"].convert("RGB")
        except Exception as e:
            print(f"[Dataset] Error loading AFHQ sample at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        img_tensor = transform(img)

        return img_tensor, ""





def download_stanford_dogs(data_dir="./data"):
    """Download Stanford Dogs dataset from HuggingFace and save to disk."""
    print("[Dataset] HuggingFace mode will load dataset dynamically during training. Skipping disk extraction to optimize startup.")
    return data_dir


def get_dataloader(data_dir="./data", batch_size=32, image_size=64, num_workers=2, sampler=None):
    """Create training dataloader."""
    dataset = AFHQDogDataset(root_dir=data_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    # Test dataset
    data_dir = download_stanford_dogs()
    loader = get_dataloader(data_dir, batch_size=4)

    for imgs, captions in loader:
        print(f"Batch images: {imgs.shape}")
        print(f"Captions: {captions}")
        break

    # Test text encoder
    enc = TextEncoder(device="cuda")
    embs = enc.encode(["a golden retriever", "a poodle puppy"])
    print(f"Text embeddings: {embs.shape}")
