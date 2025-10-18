# Repository: multi-area-rnn-wm
# Step 1 — Implement the task generator (NumPy only)
# Files in this step:
# ├─ multi_area_rnn_wm/data/wm_task.py      (new)
# └─ tests/test_task.py                      (new minimal tests)

##############################
# FILE: multi_area_rnn_wm/data/wm_task.py
##############################
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = ["TaskConfig", "SternbergWM"]

@dataclass
class TaskConfig:
    """Configuration for a Sternberg working-memory task.

    Channels layout (input):
      0                 : fixation
      1..n_identities   : identity one-hots
      last-1            : maintenance cue (on during delay)
      last              : probe cue (on during probe window)

    Outputs (targets):
      2 channels [in, out]; during probe, a ramp targets the correct choice.
    """
    n_identities: int = 5
    max_load: int = 3
    T_fix: int = 10
    T_enc: int = 50
    T_blank: int = 3
    T_maint: int = 125
    T_probe: int = 50
    dt: float = 0.02
    p_match: float = 0.5
    batch_size: int = 32
    ramp_start: float = 0.1
    ramp_end: float = 1.0

    def n_inputs(self) -> int:
        return 1 + self.n_identities + 2

    n_outputs: int = 2  # [in, out]

    def trial_length(self) -> int:
        return self.T_fix + (self.T_enc + self.T_blank) * self.max_load + self.T_maint + self.T_probe

class SternbergWM:
    """Sternberg working-memory trial generator (NumPy).

    Usage:
        cfg = TaskConfig()
        task = SternbergWM(cfg)
        batch = task.sample_batch()  # dict with X, Y, meta
    """
    def __init__(self, cfg: TaskConfig):
        assert cfg.max_load >= 1, "max_load must be >= 1"
        assert cfg.n_identities >= cfg.max_load, "n_identities should be >= max_load"
        self.cfg = cfg

    # --- indices helpers
    def _idx(self) -> Dict[str, int]:
        c = self.cfg
        FIX = 0
        ID0 = 1
        MAINT = ID0 + c.n_identities
        PROBE = MAINT + 1
        return {"FIX": FIX, "ID0": ID0, "MAINT": MAINT, "PROBE": PROBE}

    def sample_batch(self,
                     batch_size: Optional[int] = None,
                     max_load: Optional[int] = None,
                     seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        try:
            c = self.cfg
            B = batch_size or c.batch_size
            max_load = max_load or c.max_load
            assert 1 <= max_load <= c.n_identities, "max_load must be within [1, n_identities]"

            loads = np.random.randint(1, max_load + 1, size=B)
            T = c.trial_length()
            X = np.zeros((B, T, c.n_inputs()), dtype=np.float32)
            Y = np.zeros((B, T, c.n_outputs), dtype=np.float32)

            idx = self._idx()

            probe_id = np.zeros(B, dtype=np.int32)
            probe_in = np.zeros(B, dtype=np.int32)
            chosen_list: List[np.ndarray] = []

            for b in range(B):
                t = 0
                # fixation
                X[b, t:t+c.T_fix, idx["FIX"]] = 1.0
                t += c.T_fix

                # choose the set for this trial
                pool = np.random.permutation(c.n_identities)
                chosen = pool[: loads[b]]
                chosen_list.append(chosen)

                # sequential encoding (pad with blanks up to max_load slots)
                for k in range(c.max_load):
                    if k < loads[b]:
                        idk = int(chosen[k])
                        X[b, t:t+c.T_enc, idx["ID0"] + idk] = 1.0
                    t += c.T_enc
                    t += c.T_blank

                # maintenance
                X[b, t:t+c.T_maint, idx["MAINT"]] = 1.0
                t += c.T_maint

                # probe: pick in-set or lure
                is_in = (np.random.rand() < c.p_match)
                if is_in:
                    pid = int(np.random.choice(chosen))
                    probe_in[b] = 1
                else:
                    lure_pool = [i for i in range(c.n_identities) if i not in set(chosen.tolist())]
                    pid = int(np.random.choice(lure_pool))
                    probe_in[b] = 0
                probe_id[b] = pid

                # probe channels
                X[b, t:t+c.T_probe, idx["PROBE"]] = 1.0
                X[b, t:t+c.T_probe, idx["ID0"] + pid] = 1.0

                # targets: ramp on correct choice channel
                ramp = np.linspace(c.ramp_start, c.ramp_end, c.T_probe, dtype=np.float32)
                Y[b, t:t+c.T_probe, 0 if is_in else 1] = ramp

            meta = {
                "loads": loads,
                "probe_id": probe_id,
                "probe_in": probe_in,
                "chosen": chosen_list,
                "idx": idx,
                "T_total": T,
            }
            return {"X": X, "Y": Y, "meta": meta}
        finally:
            if seed is not None:
                np.random.set_state(rng_state)

    # --- quick viz helper (optional; no deps beyond NumPy)
    def summarize(self, batch: Dict[str, np.ndarray]) -> Dict[str, int]:
        X, Y = batch["X"], batch["Y"]
        return {
            "B": int(X.shape[0]),
            "T": int(X.shape[1]),
            "I": int(X.shape[2]),
            "O": int(Y.shape[2]),
        }


if __name__ == "__main__":
    # Manual smoke run
    cfg = TaskConfig()
    task = SternbergWM(cfg)
    out = task.sample_batch(batch_size=4, seed=123)
    print("summary:", task.summarize(out))
    print("loads:", out["meta"]["loads"])  # e.g., array([2,1,3,2])
