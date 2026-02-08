
import torch
import torch.nn as nn

class PointROPE(nn.Module):
    def __init__(self, freq=100.0):
        super().__init__()
        self.freq = freq
        
    def forward(self, q, pos):
        """
        Mock forward pass for PointROPE.
        Just returns q as is (identity).
        """
        return q
