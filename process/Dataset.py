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
        eos_token_id: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize the SequenceDataset.

        Args:
            context_window (int): Number of context tokens in each sample.
            target_window (int): Number of target tokens in each sample.
            data (List[List[int]]): The tokenized data to process into sequences.
            pad_token_id (int): Token ID used for padding.
            eos_token_id (int): Token ID used for EOS (End Of Sequence).
            verbose (bool): Whether to display initialization details.
        """
        self.context_window = context_window
        self.target_window = target_window
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.verbose = verbose

        # Preprocess data: Add EOS token and pad/truncate
        self.data = []
        for sequence in data:
            sequence = sequence[:context_window + target_window - 1]  # Enforce max length
            sequence += [eos_token_id]  # Append EOS
            self.data.append(sequence)

        # Build index for sequences
        self.sample_index = []
        for seq_idx, seq in enumerate(self.data):
            if len(seq) >= context_window + target_window:  # Ensure enough tokens for one sample
                for i in range(len(seq) - context_window - target_window + 1):
                    self.sample_index.append((seq_idx, i))

        if self.verbose:
            print(f"SequenceDataset initialized with {len(self.sample_index)} samples.")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Context and target sequences as tensors.
        """
        seq_idx, start_idx = self.sample_index[idx]
        sequence = self.data[seq_idx]

        # Extract context and target
        context = sequence[start_idx : start_idx + self.context_window]
        target = sequence[start_idx + self.context_window : start_idx + self.context_window + self.target_window]

        # Pad context and target if necessary
        context = context + [self.pad_token_id] * (self.context_window - len(context))
        target = target + [self.pad_token_id] * (self.target_window - len(target))

        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of samples into padded tensors.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): A batch of context-target pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batched and padded context and target tensors.
        """
        contexts, targets = zip(*batch)
        context_padded = torch.stack(contexts)
        target_padded = torch.stack(targets)
        return context_padded, target_padded
