import torch
import torch.nn as nn

class Dummy(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim) 
        )
        self.out_dim = out_dim 
        
    def forward(self, x=None):
        if x is not None:
            out = self.net(x)
            return out 

        return torch.zeros((B, self.out_dim))
        