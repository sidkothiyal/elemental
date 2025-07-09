"""WikiText paragraphs dataset."""
import pathlib
import random
from typing import Dict
from typing import Optional
from typing import Union

import torch
from torch.utils import data
from transformers import AutoTokenizer

from dataset.cache_decorator import cache_getitem


class WikiTextDataset(data.Dataset):
    """WikiText paragraphs dataset."""

    def __init__(
        self,
        dataset_dir: Union[str, pathlib.Path],
        split: str,
        seed: int = 42,
        max_length: int = 512,
        pretrained_tokenizer: str = "gpt2",
        num_samples: Optional[int] = None,
    ):
        """WikiText dataset constructor.

        Args:
            dataset_dir: Path to the directory containing paragraph files.
            split: Split to use ('train' or 'val').
            seed: Random seed for reproducible train/val split.
            max_length: Maximum sequence length for tokenization.
            pretrained_tokenizer: Pretrained tokenizer to use.
            num_samples: Number of samples to use. If None, use all samples.
        """
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.split = split
        self.seed = seed
        self.max_length = max_length
        self.num_samples = num_samples

        # Initialize GPT-2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Collect all paragraph files
        self.paragraph_files = list(self.dataset_dir.glob("paragraph_*.txt"))
        self.paragraph_files.sort()  # Ensure consistent ordering

        # Split data into train/val with fixed seed
        random.seed(self.seed)
        random.shuffle(self.paragraph_files)

        split_idx = int(0.8 * len(self.paragraph_files))
        if self.split == "train":
            self.paragraph_files = self.paragraph_files[:split_idx]
        elif self.split == "val":
            self.paragraph_files = self.paragraph_files[split_idx:]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {self.split}")

        print(f"WikiText {self.split} dataset: {len(self.paragraph_files)} paragraphs")

    def __len__(self):
        """Get the length of the dataset."""
        if self.num_samples:
            return self.num_samples
        return len(self.paragraph_files)

    @cache_getitem(cache_dir="cache/wikitext_dataset")
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            idx: Index of the item to get.

        Returns:
            Dictionary containing tokenized input_ids and attention_mask.
        """
        paragraph_file = self.paragraph_files[idx]

        # Read paragraph text
        with open(paragraph_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input": encoding["input_ids"].squeeze(0),
            "target": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


if __name__ == "__main__":
    from tqdm import tqdm

    # Test the dataset
    dataset_dir = "data/wikitext/paragraphs"

    print("Creating train dataset...")
    train_dataset = WikiTextDataset(
        dataset_dir=dataset_dir, split="train", seed=42, max_length=4096
    )

    print("Creating val dataset...")
    val_dataset = WikiTextDataset(dataset_dir=dataset_dir, split="val", seed=42, max_length=4096)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Total paragraphs: {len(train_dataset) + len(val_dataset)}")

    # Test a few samples
    print("\nTesting train samples:")
    for i in range(3):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  input_ids shape: {sample['input'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  input_ids: {sample['input'][:20].tolist()}...")
        print()

    print("Testing val samples:")
    for i in range(3):
        sample = val_dataset[i]
        print(f"Sample {i}:")
        print(f"  input_ids shape: {sample['input'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  input_ids: {sample['input'][:20].tolist()}...")
        print()

    # Test dataloader
    print("Testing dataloader...")
    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    for batch_idx, batch in enumerate(tqdm(train_dataloader, total=5)):
        if batch_idx >= 5:  # Only test first 5 batches
            break
        print(f"Batch {batch_idx}:")
        print(f"  input_ids shape: {batch['input'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print()

    print("Dataset verification completed successfully!")
