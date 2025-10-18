##############################
# FILE: tests/test_torch_smoke.py
##############################
import os
import numpy as np
from multi_area_rnn_wm.data.wm_task import TaskConfig, SternbergWM
from multi_area_rnn_wm.models.rnn_torch import TORCH_AVAILABLE, SimpleRNNTorch

import pytest

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_torch_forward_and_loss():
    import torch
    cfg = TaskConfig(batch_size=16, T_probe=20)
    task = SternbergWM(cfg)
    batch = task.sample_batch(seed=0)

    model = SimpleRNNTorch(input_dim=cfg.n_inputs(), hidden_dim=64, output_dim=2)
    X = torch.from_numpy(batch["X"]) ; Y = torch.from_numpy(batch["Y"]) 
    logits, H = model(X)
    assert logits.shape == Y.shape
    # simple MSE on softmax
    p = torch.softmax(logits, dim=-1)
    loss = ((p - Y)**2).mean()
    assert torch.isfinite(loss)
