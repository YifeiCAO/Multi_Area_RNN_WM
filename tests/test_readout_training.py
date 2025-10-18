##############################
# FILE: tests/test_readout_training.py
##############################
import numpy as np
from multi_area_rnn_wm.data.wm_task import TaskConfig, SternbergWM
from multi_area_rnn_wm.models.rnn_core import RNNConfig, SimpleRNNNumPy, train_readout_sgd

def test_train_readout_sgd_monotonic_drop():
    cfg = TaskConfig(n_identities=6, max_load=3, batch_size=256, T_probe=30)
    task = SternbergWM(cfg)
    batch = task.sample_batch(seed=2025)
    X, Y = batch["X"], batch["Y"]

    rcfg = RNNConfig(input_dim=cfg.n_inputs(), hidden_dim=64, output_dim=2, nonlinearity="tanh")
    rnn = SimpleRNNNumPy(rcfg)

    T = cfg.trial_length()
    s = slice(T - cfg.T_probe, T)
    _, H = rnn.forward(X)
    Hp = H[:, s, :].reshape(-1, rcfg.hidden_dim)
    Yp = Y[:, s, :].reshape(-1, 2)

    W, b, losses = train_readout_sgd(Hp, Yp, epochs=30, lr=5e-2, batch_size=512)
    # 换上学到的读出
    rnn.set_readout(W, b)

    # 训练期间的 loss 应该总体下降（允许小幅波动，这里检查首末差距）
    assert losses[0] > losses[-1]
