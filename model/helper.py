import torch


def causal_mask(seq_len, num_latents, device=None):
    """
    Creates a causal mask for self-attention with added latent dimensions.

    Args:
        seq_len (int): Length of the input sequence.
        num_latents (int): Number of latent vectors added to the keys and values.
        device (torch.device, optional): Device for the mask tensor.

    Returns:
        Tensor: A causal mask of shape (1, 1, seq_len, seq_len + num_latents).
    """
    seq_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    seq_mask = seq_mask.masked_fill(seq_mask == 1, float("-inf"))

    latent_mask = torch.zeros(seq_len, num_latents, device=device)

    full_mask = torch.cat([seq_mask, latent_mask], dim=-1)

    return full_mask.unsqueeze(0).unsqueeze(0)

def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
