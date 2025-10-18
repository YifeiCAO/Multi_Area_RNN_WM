##############################
# FILE: multi_area_rnn_wm/models/rnn_core.py
##############################
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

__all__ = ["RNNConfig", "SimpleRNNNumPy", "ridge_fit_readout", "ridge_fit_readout_with_bias", "train_readout_sgd"]

@dataclass
class RNNConfig:
    input_dim: int
    hidden_dim: int = 64
    output_dim: int = 2
    nonlinearity: str = "tanh"  # or "relu"
    weight_scale: float = 0.1
    seed: int = 1337

class SimpleRNNNumPy:
    """A minimal NumPy RNN (discrete-time) with linear readout.

    h_{t+1} = f(Wxh x_t + Whh h_t + b)
    y_t     = Wout h_t + bout

    Notes:
      • Pure NumPy, no autodiff. For learning, we recommend ridge-fitting Wout
        using hidden states as features (see `ridge_fit_readout`).
      • Or use `train_readout_sgd` for an *iterative* training loop with a loss curve.
    """
    def __init__(self, cfg: RNNConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.Wxh = rng.normal(0.0, cfg.weight_scale, size=(cfg.hidden_dim, cfg.input_dim))
        self.Whh = rng.normal(0.0, cfg.weight_scale, size=(cfg.hidden_dim, cfg.hidden_dim))
        # Spectral scaling for stability
        u, s, vh = np.linalg.svd(self.Whh, full_matrices=False)
        self.Whh = (self.Whh / (np.max(s) + 1e-8)) * 0.9
        self.bh  = np.zeros((cfg.hidden_dim,), dtype=float)
        self.Wout = rng.normal(0.0, cfg.weight_scale, size=(cfg.output_dim, cfg.hidden_dim))
        self.bout = np.zeros((cfg.output_dim,), dtype=float)

    def _f(self, z: np.ndarray) -> np.ndarray:
        if self.cfg.nonlinearity == "relu":
            return np.maximum(z, 0.0)
        return np.tanh(z)

    def forward(self, X: np.ndarray, h0: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run the RNN.
        Args:
          X: [B, T, I]
          h0: [B, H] or None
        Returns:
          Y: [B, T, O]
          H: [B, T, H]
        """
        B, T, I = X.shape
        Hdim, O = self.cfg.hidden_dim, self.cfg.output_dim
        if h0 is None:
            h = np.zeros((B, Hdim), dtype=float)
        else:
            h = h0.copy()
        H = np.zeros((B, T, Hdim), dtype=float)
        Y = np.zeros((B, T, O), dtype=float)
        for t in range(T):
            xt = X[:, t, :]  # [B, I]
            h = self._f(xt @ self.Wxh.T + h @ self.Whh.T + self.bh)
            H[:, t, :] = h
            Y[:, t, :] = h @ self.Wout.T + self.bout
        return Y, H

    def set_readout(self, Wout: np.ndarray, b: Optional[np.ndarray] = None):
        assert Wout.shape == self.Wout.shape
        self.Wout[...] = Wout
        if b is not None:
            assert b.shape == self.bout.shape
            self.bout[...] = b


def ridge_fit_readout(H: np.ndarray, Y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    """Fit a linear readout with ridge regression (no bias).
    Args:
      H: [N, H] features (stack batch×time or selected windows)
      Y: [N, O] targets
      alpha: L2 regularization
    Returns:
      Wout: [O, H]
    """
    HtH = H.T @ H
    O = Y.shape[1]
    I = np.eye(HtH.shape[0])
    W = np.linalg.solve(HtH + alpha * I, H.T @ Y)  # [H, O]
    return W.T  # [O, H]


def ridge_fit_readout_with_bias(H: np.ndarray, Y: np.ndarray, alpha: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge with bias term via feature augmentation. Returns (Wout, b)."""
    N, Hdim = H.shape
    H1 = np.concatenate([H, np.ones((N, 1))], axis=1)  # [N, H+1]
    HtH = H1.T @ H1
    I = np.eye(HtH.shape[0])
    Wb = np.linalg.solve(HtH + alpha * I, H1.T @ Y)  # [H+1, O]
    Wout = Wb[:-1, :].T
    b = Wb[-1, :].T
    return Wout, b


def train_readout_sgd(H: np.ndarray, Y: np.ndarray, epochs: int = 200, lr: float = 1e-2, batch_size: int = 1024, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Iterative SGD on a linear readout with bias (NumPy).

    Args:
      H: [N, H] features
      Y: [N, O] targets
      epochs: number of passes over data
      lr: learning rate
      batch_size: mini-batch size
    Returns:
      (Wout, b, losses)
    """
    N, Hdim = H.shape
    O = Y.shape[1]
    rng = np.random.default_rng(1234)
    W = rng.normal(0.0, 0.01, size=(O, Hdim))
    b = np.zeros((O,), dtype=float)
    losses: List[float] = []
    idx_all = np.arange(N)
    for ep in range(epochs):
        if shuffle:
            rng.shuffle(idx_all)
        loss_ep = 0.0
        for i in range(0, N, batch_size):
            idx = idx_all[i:i+batch_size]
            Hb = H[idx]          # [B, H]
            Yb = Y[idx]          # [B, O]
            # forward
            Yhat = Hb @ W.T + b  # [B, O]
            err = Yhat - Yb
            loss = (err**2).mean()
            loss_ep += float(loss) * len(idx)
            # grads
            dW = (2.0/len(idx)) * (err.T @ Hb)    # [O, H]
            db = (2.0/len(idx)) * err.sum(axis=0) # [O]
            # sgd step
            W -= lr * dW
            b -= lr * db
        losses.append(loss_ep / N)
    return W, b, losses

