##############################
# FILE: tests/test_rnn_forward.py
##############################
import numpy as np
from multi_area_rnn_wm.data.wm_task import TaskConfig, SternbergWM
from multi_area_rnn_wm.models.rnn_core import RNNConfig, SimpleRNNNumPy, ridge_fit_readout


def test_forward_shapes():
    cfg = TaskConfig(n_identities=5, max_load=3, batch_size=4)
    task = SternbergWM(cfg)
    batch = task.sample_batch(seed=0)
    X, Y = batch["X"], batch["Y"]

    rcfg = RNNConfig(input_dim=cfg.n_inputs(), hidden_dim=32, output_dim=cfg.n_outputs)
    rnn = SimpleRNNNumPy(rcfg)

    Yhat, H = rnn.forward(X)
    assert Yhat.shape == Y.shape
    assert H.shape == (cfg.batch_size, cfg.trial_length(), rcfg.hidden_dim)


def test_ridge_fit_improves_probe_accuracy():
    """Ridge-fitting readout on probe windows should beat random baseline."""
    cfg = TaskConfig(n_identities=6, max_load=3, batch_size=128, T_probe=30)
    task = SternbergWM(cfg)

    # train batch
    btr = task.sample_batch(seed=1)
    Xtr, Ytr = btr["X"], btr["Y"]

    rcfg = RNNConfig(input_dim=cfg.n_inputs(), hidden_dim=64, output_dim=2)
    rnn = SimpleRNNNumPy(rcfg)

    # collect features on probe window only
    T = cfg.trial_length()
    s = slice(T - cfg.T_probe, T)
    _, Htr = rnn.forward(Xtr)
    Hp = Htr[:, s, :].reshape(-1, rcfg.hidden_dim)       # [B*Tprobe, H]
    Yp = Ytr[:, s, :].reshape(-1, 2)                     # [B*Tprobe, 2]

    # fit readout
    Wout = ridge_fit_readout(Hp, Yp, alpha=1e-2)
    rnn.set_readout(Wout)

    # eval on new batch
    bte = task.sample_batch(seed=2)
    Xte, Yte = bte["X"], bte["Y"]
    Yhat, _ = rnn.forward(Xte)
    # simple accuracy over the probe window based on mean DV
    s = slice(T - cfg.T_probe, T)
    y_pred = Yhat[:, s, :].mean(axis=1).argmax(axis=1)
    y_true = Yte[:, s, :].mean(axis=1).argmax(axis=1)
    acc = (y_pred == y_true).mean()

    # should be better than random 0.5 in most runs
    assert acc >= 0.55