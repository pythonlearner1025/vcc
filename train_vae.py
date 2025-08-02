#!/usr/bin/env python3
"""Standalone training script for the ESM-aware VAE.

The script intentionally mirrors the structure of `train.py` so users familiar
with the diffusion training loop feel at home.  It is **not** integrated with
Hydra for now to keep the diff small – plain `argparse` is sufficient and easy

to refactor later.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from dataset.umi_loader import UMITrainLoader, umi_collate
from models.vae import VAE


# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# cosine LR helper ----------------------------------------------------------
# ---------------------------------------------------------------------------
import math

def cosine_lr_schedule(optimizer, step: int, total_steps: int, base_lr: float = 1e-4, warmup_steps: int = 500):
    """Cosine schedule with linear warm-up (matches models.diffusion version)."""
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr

# ---------------------------------------------------------------------------
# training loop -------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optim, device, beta: float, global_step: int, meter_counts: torch.Tensor, interval: int, total_steps: int, base_lr: float, warmup_steps: int):
    model.train()
    losses = []  # per-epoch running loss
    for batch_idx, batch in enumerate(loader):
        x = batch.to(device)
        mu, log_var = model.encode(x)
        z = model.reparameterise(mu, log_var)
        recon = model.decode(z)
        # --- token tracking for perplexity ---
        with torch.no_grad():
            tokens = model.tokenise(z.detach())  # (B, P)
            meter_counts += torch.bincount(tokens.flatten(), minlength=model.tokenizer.codebook_size).to(device)
        recon_loss = model.recon_loss(x, recon, reduction="mean")
        kl = model.kl_loss(mu, log_var)
        loss = recon_loss + beta * kl

        optim.zero_grad()
        # learning-rate step before backward matches diffusion training style
        lr = cosine_lr_schedule(optim, global_step, total_steps, base_lr, warmup_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        losses.append(loss.item())

        # logging ----------------------------------------------------------------
        if global_step % interval == 0:
            probs = meter_counts / meter_counts.sum().clamp(min=1)
            perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item()
        
            avg_loss = np.mean(losses[-100:] if len(losses) > 100 else losses)
            print(f"Batch[{batch_idx+1:4d}/{len(loader):4d}] | "
                  f"Step {global_step:6d} | Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | PPL: {perplexity:.1f} | LR: {lr:.2e}")

            wandb.log({
                "train/loss": loss.item(),
                "train/recon": recon_loss.item(),
                "train/kl": kl.item(),
                "step": global_step,
            })
        global_step += 1
    return np.mean(losses), global_step


# ---------------------------------------------------------------------------
# main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VAE on raw UMI counts")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--esm_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--patch_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--normalize", type=bool, default=0)
    parser.add_argument("--enc_hidden_dim", type=int, default=4096)
    parser.add_argument("--perplexity_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # dataset ----------------------------------------------------------------
    ds = UMITrainLoader(args.data_dir, normalize=args.normalize)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=umi_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # model ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(
        n_genes=ds.n_genes,
        esm_emb_path=args.esm_path,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        patch_dim=args.patch_dim,
        enc_hidden_dim=args.enc_hidden_dim,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # optimiser --------------------------------------------------------------
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # wandb ------------------------------------------------------------------
    run = wandb.init(
        project="vcc-vae",
        config=vars(args),
        name=f"vae_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    # training ---------------------------------------------------------------
    global_step = 0
    meter_counts = torch.zeros(args.codebook_size, device=device)
    total_steps = args.epochs * len(loader)
    for epoch in range(args.epochs):
        avg_loss, global_step = train_epoch(
            model, loader, optim, device, args.beta, global_step, meter_counts, args.perplexity_interval, total_steps, args.lr, 500
        )
        print(f"Epoch {epoch}/{args.epochs}  |  avg loss = {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})

    # save -------------------------------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "vae_ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "codebook": model.tokenizer.codebook.weight}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path.relative_to(Path.cwd())}")
    run.finish()


if __name__ == "__main__":
    main()
