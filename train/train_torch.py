##############################
# FILE: multi_area_rnn_wm/train/train_torch.py
##############################
from __future__ import annotations
import argparse, math, random
import numpy as np

from multi_area_rnn_wm.data.wm_task import TaskConfig, SternbergWM
from multi_area_rnn_wm.models.rnn_torch import TORCH_AVAILABLE, SimpleRNNTorch

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def numpy_batch_to_torch(batch, device):
    X = torch.from_numpy(batch["X"]).to(device)
    Y = torch.from_numpy(batch["Y"]).to(device)
    return X, Y, batch["meta"]


def decision_loss(logits, targets):
    # MSE between softmax(logits) and analog targets
    p = F.softmax(logits, dim=-1)
    return F.mse_loss(p, targets)


def eval_probe_accuracy(logits, targets, T_probe):
    B, T, _ = logits.shape
    s = slice(T - T_probe, T)
    p = F.softmax(logits[:, s, :], dim=-1).mean(dim=1)  # [B,2]
    y = targets[:, s, :].mean(dim=1).argmax(dim=-1)
    pred = p.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train(steps=5000, device="cpu", hidden_dim=128, lr=1e-3, seed=1337,
          log_every=100, eval_every=500):
    assert TORCH_AVAILABLE, "PyTorch is not available in this environment."
    set_seed(seed)

    # Task
    tcfg = TaskConfig(batch_size=128, n_identities=6, max_load=3, T_probe=30)
    task = SternbergWM(tcfg)

    # Model
    model = SimpleRNNTorch(input_dim=tcfg.n_inputs(), hidden_dim=hidden_dim, output_dim=2, nonlinearity="tanh")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 1e9

    for step in range(1, steps+1):
        model.train()
        batch = task.sample_batch()
        X, Y, meta = numpy_batch_to_torch(batch, device)

        opt.zero_grad()
        logits, _ = model(X)
        loss = decision_loss(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0:
            acc = eval_probe_accuracy(logits.detach(), Y, tcfg.T_probe)
            print(f"step {step:6d} | train loss {loss.item():.4f} | probe-acc ~ {acc:.3f}")

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                batch = task.sample_batch()
                Xv, Yv, _ = numpy_batch_to_torch(batch, device)
                logits, _ = model(Xv)
                val = decision_loss(logits, Yv).item()
                acc = eval_probe_accuracy(logits, Yv, tcfg.T_probe)
            print(f"[eval] loss {val:.4f} | acc {acc:.3f}")
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "cfg": {"input": tcfg.n_inputs(), "hidden": hidden_dim, "out": 2}}, "simple_rnn_torch.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train"], help="run training")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.cmd == "train":
        train(steps=args.steps, device=args.device, hidden_dim=args.hidden, lr=args.lr, seed=args.seed)

if __name__ == "__main__":
    main()

