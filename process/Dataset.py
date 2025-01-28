import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class SequenceDataset(Dataset):
    def __init__(
        self,
        context_window: int,
        target_window: int,
        data: List[int],
        verbose: bool = False,
    ):
        """
        Initialize the SequenceDataset.

        Args:
            context_window (int): Number of context tokens in each sample.
            target_window (int): Number of target tokens in each sample.
            data (List[int]): The raw data to process into sequences.
            verbose (bool): Whether to display a progress bar during initialization.
        """
        self.context_window = context_window
        self.target_window = target_window
        self.data = data
        self.verbose = verbose

        self.total_sequences = (
            len(self.data) - self.context_window - self.target_window + 1
        )

        if self.verbose:
            print(f"SequenceDataset initialized with {self.total_sequences} samples.")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.total_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazily retrieve a single (context, target) pair by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The context and target tensors.
        """
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(
                f"Index {idx} is out of range for dataset with {self.total_sequences} samples."
            )

        context = self.data[idx : idx + self.context_window]
        target = self.data[
            idx + self.context_window : idx + self.context_window + self.target_window
        ]

        return torch.tensor(context, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )
