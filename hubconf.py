from typing import Literal

import torch

from src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
from src.phi import Phi

dependencies = ['torch']


def searle(backbone: Literal['ViT-B/32', 'ViT-L/14']):
    """
    Load textual inversion network Phi
    """
    if backbone == 'ViT-B/32':
        input_dim = 512
        hidden_dim = 2048
        output_dim = 512
    elif backbone == 'ViT-L/14':
        input_dim = 768
        hidden_dim = 3072
        output_dim = 768
    else:
        raise ValueError(f'Unknown backbone {backbone}')

    phi = Phi(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.5)
    phi = phi.eval()

    checkpoint_url = f"https://github.com/miccunifi/SEARLE/releases/download/weights/SEARLE_{backbone.replace('/', '')}.pt"
    phi.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')[phi.__class__.__name__])

    return phi, encode_with_pseudo_tokens
