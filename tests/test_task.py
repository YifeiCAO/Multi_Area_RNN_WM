# Repository: multi-area-rnn-wm
# Step 1 — Implement the task generator (NumPy only)
# Files in this step:
# ├─ multi_area_rnn_wm/data/wm_task.py      (new)
# └─ tests/test_task.py                      (new minimal tests)

##############################
# FILE: tests/test_task.py
##############################
import numpy as np
from multi_area_rnn_wm.data.wm_task import TaskConfig, SternbergWM

def test_shapes_and_types():
    cfg = TaskConfig(n_identities=6, max_load=3, batch_size=5)
    t = SternbergWM(cfg)
    batch = t.sample_batch(seed=42)
    X, Y, meta = batch["X"], batch["Y"], batch["meta"]
    assert X.shape[0] == cfg.batch_size
    assert X.shape[2] == cfg.n_inputs()
    assert Y.shape == (cfg.batch_size, cfg.trial_length(), cfg.n_outputs)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    assert meta["T_total"] == cfg.trial_length()


def test_probe_targets_are_ramps():
    cfg = TaskConfig(T_probe=20)
    t = SternbergWM(cfg)
    batch = t.sample_batch(seed=0)
    X, Y, meta = batch["X"], batch["Y"], batch["meta"]
    idx = meta["idx"]
    # find the start of probe: last 20 time steps
    T = cfg.trial_length()
    probe_slice = slice(T - cfg.T_probe, T)
    # ramps should be monotonic on exactly one of the two output channels for each trial
    ramps_ok = 0
    for b in range(cfg.batch_size):
        y = Y[b, probe_slice, :]
        # exactly one channel should be positive ramp
        pos_channels = (y.sum(axis=0) > 0).astype(int)
        assert pos_channels.sum() == 1
        # monotonic non-decreasing on the active channel
        k = int(np.argmax(pos_channels))
        diffs = np.diff(y[:, k])
        if np.all(diffs >= -1e-6):
            ramps_ok += 1
    assert ramps_ok == cfg.batch_size


def test_encoding_length_scales_with_max_load():
    cfg = TaskConfig(max_load=3, T_enc=7, T_blank=2)
    t = SternbergWM(cfg)
    # use seed to reproduce loads
    b1 = t.sample_batch(seed=1)
    b2 = t.sample_batch(seed=2)
    # trial length is fixed by cfg.max_load (encoding budget for max slots)
    assert b1["X"].shape[1] == cfg.trial_length()
    assert b2["X"].shape[1] == cfg.trial_length()
