import torch
import torch.nn as nn

class Projector(nn.Module):
    """
    Lightweight adapter that maps HuBERT embeddings (768-D) to
    the 4800-D space expected by EDGE (Jukebox-style).
    Architecture: Linear(768→1536) + GELU + Linear(1536→4800)
    """
    def __init__(self, in_dim=768, hidden=1536, out_dim=4800):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # Orthogonal init helps maintain variance and stability at start.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)