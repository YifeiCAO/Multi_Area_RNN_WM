##############################
# FILE: multi_area_rnn_wm/models/rnn_torch.py
##############################
from __future__ import annotations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # keep import-time safe on envs without torch
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

class SimpleRNNTorch(nn.Module):
    """Minimal RNN with manual time loop.
    Inputs:  [B, T, I]
    Outputs: logits [B, T, O], states [B, T, H]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2,
                 nonlinearity: str = "tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cell = nn.RNNCell(input_dim, hidden_dim, nonlinearity=nonlinearity)
        self.readout = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, X: "torch.Tensor", h0: "torch.Tensor | None" = None):
        B, T, I = X.shape
        H = torch.zeros(B, T, self.hidden_dim, device=X.device)
        Y = torch.zeros(B, T, self.output_dim, device=X.device)
        h = torch.zeros(B, self.hidden_dim, device=X.device) if h0 is None else h0
        for t in range(T):
            h = self.cell(X[:, t, :], h)
            H[:, t, :] = h
            Y[:, t, :] = self.readout(h)
        return Y, H
    
class GRUTorch(nn.Module):
    """GRU: Inputs [B,T,I] -> logits [B,T,O], states [B,T,H]"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)   # ✔ batch_first
        self.readout = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, X: "torch.Tensor", h0: "torch.Tensor|None" = None):
        # X: [B,T,I]; h0(optional): [1,B,H]
        out, h_n = self.rnn(X, h0)            # out: [B,T,H]
        logits = self.readout(out)            # [B,T,O]
        return logits, out                    # 与你原先的返回一致
