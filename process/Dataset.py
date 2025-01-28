import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class SequenceDataset(Dataset):
    def __init__(
        self,
        context_window: int,
        target_window: int,
        data: List[List[int]],
        pad_token_id: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize the SequenceDataset.

        Args:
            context_window (int): Number of context tokens in each sample.
            target_window (int): Number of target tokens in each sample.
            data (List[List[int]]): The tokenized data to process into sequences.
            pad_token_id (int): Token ID used for padding.
            verbose (bool): Whether to display initialization details.
        """
        self.context_window = context_window
        self.target_window = target_window
        self.data = data
        self.pad_token_id = pad_token_id
        self.verbose = verbose  

        self.total_sequences = sum(
            max(0, len(seq) - context_window - target_window + 1) for seq in data
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
        cumulative_idx = 0

        # Locate the correct sequence in the data
        for sequence in self.data:
            num_sequences = max(
                0, len(sequence) - self.context_window - self.target_window + 1
            )
            if idx < cumulative_idx + num_sequences:
                sequence_idx = idx - cumulative_idx
                start = sequence_idx
                context = sequence[start : start + self.context_window]
                target = sequence[
                    start + self.context_window : start
                    + self.context_window
                    + self.target_window
                ]

                # Pad if needed
                context = context + [self.pad_token_id] * (
                    self.context_window - len(context)
                )
                target = target + [self.pad_token_id] * (
                    self.target_window - len(target)
                )

                return torch.tensor(context, dtype=torch.long), torch.tensor(
                    target, dtype=torch.long
                )

            cumulative_idx += num_sequences

        raise IndexError("Index out of range.")
