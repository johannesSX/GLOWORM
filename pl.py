"""
pl.py — Training loop for Graph Cycle-VAE
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from cycle_vae import GraphCycleVAE, count_params


def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    totals = {'loss': 0, 'recon': 0, 'cycle': 0, 'kl': 0,
              'recon_a': 0, 'recon_b': 0, 'cycle_a': 0, 'cycle_b': 0}
    n = 0
    use_amp = device.type == 'cuda'

    pbar = tqdm(loader, desc=f"  Train {epoch}/{total_epochs}", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for batch in pbar:
        batch = batch.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch)
            kld_weight = 1.0 / batch.x.size(0)
            loss_dict = model.compute_loss(out, kld_weight)
            loss = loss_dict['loss']

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals['loss'] += loss.item()
        totals['recon'] += loss_dict['recon_loss'].item()
        totals['cycle'] += loss_dict['cycle_loss'].item()
        totals['kl'] += loss_dict['kl_loss'].item()
        totals['recon_a'] += loss_dict['recon_a'].item()
        totals['recon_b'] += loss_dict['recon_b'].item()
        totals['cycle_a'] += loss_dict['cycle_a'].item()
        totals['cycle_b'] += loss_dict['cycle_b'].item()
        n += 1

        pbar.set_postfix_str(
            f"L={loss.item():.4f} "
            f"Ra={totals['recon_a']/n:.4f} Rb={totals['recon_b']/n:.4f} "
            f"Ca={totals['cycle_a']/n:.4f} Cb={totals['cycle_b']/n:.4f} "
            f"KL={totals['kl']/n:.4f}")

    pbar.close()
    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    totals = {'loss': 0, 'recon': 0, 'cycle': 0, 'kl': 0,
              'recon_a': 0, 'recon_b': 0, 'cycle_a': 0, 'cycle_b': 0}
    n = 0
    use_amp = device.type == 'cuda'

    for batch in tqdm(loader, desc="  Val", leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
        batch = batch.to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(batch)
            loss_dict = model.compute_loss(out, 1.0 / batch.x.size(0))

        v = loss_dict['loss'].item()
        if not (np.isnan(v) or np.isinf(v)):
            totals['loss'] += v
            totals['recon'] += loss_dict['recon_loss'].item()
            totals['cycle'] += loss_dict['cycle_loss'].item()
            totals['kl'] += loss_dict['kl_loss'].item()
            totals['recon_a'] += loss_dict['recon_a'].item()
            totals['recon_b'] += loss_dict['recon_b'].item()
            totals['cycle_a'] += loss_dict['cycle_a'].item()
            totals['cycle_b'] += loss_dict['cycle_b'].item()
            n += 1

    n = max(n, 1)
    return {k: v / n for k, v in totals.items()}


def run_train_loop(args, train_loader, val_loader, device):
    if device.type == 'cuda':
        print(f"  BF16 autocast on {torch.cuda.get_device_name()}")

    from torch.utils.tensorboard import SummaryWriter
    tb_dir = getattr(args, 'tensorboard_dir', 'runs/cycle_vae')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"  TensorBoard: {tb_dir}")

    print("\n=== Graph Cycle-VAE ===")
    num_mod = args.num_features - args.num_morpho
    split_mode = getattr(args, 'split_mode', 'morpho_mod')
    model = GraphCycleVAE(
        num_features=args.num_features,
        split_mode=split_mode,
        num_morpho=args.num_morpho,
        indices_a=getattr(args, 'indices_a', None),
        indices_b=getattr(args, 'indices_b', None),
        dim_h=args.dim_h, dim_pe=args.dim_pe, latent_dim=args.latent_dim,
        num_pool_levels=args.num_pool_levels, pool_ratio=args.pool_ratio,
        gnn_layers_per_level=args.gnn_layers_per_level,
        beta=args.beta, lambda_cycle=args.lambda_cycle,
        dropout=args.dropout, act='relu',
        num_atlas_regions=args.num_atlas_regions,
        dim_atlas=args.dim_atlas,
        edge_weight=getattr(args, 'edge_weight', 0.0),
    ).to(device)

    resume = getattr(args, 'resume_checkpoint', None)
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device, weights_only=True), strict=False)
        print(f"  Resumed from: {resume}")

    total_p, train_p = count_params(model)
    print(f"  Parameters: {train_p:,} trainable / {total_p:,} total")
    print(f"  Split: {split_mode} — "
          f"group_a={model.indices_a} ({len(model.indices_a)}d) ↔ "
          f"group_b={model.indices_b} ({len(model.indices_b)}d)")
    print(f"  dim_h={args.dim_h}, latent={args.latent_dim}, "
          f"pool={args.num_pool_levels}, beta={args.beta}, λ_cycle={args.lambda_cycle}, "
          f"edge_weight={getattr(args, 'edge_weight', 1.0)}")

    writer.add_text("hparams", (
        f"morpho={args.num_morpho}, mod={num_mod}, dim_h={args.dim_h}, "
        f"latent={args.latent_dim}, pool={args.num_pool_levels}, "
        f"beta={args.beta}, lambda_cycle={args.lambda_cycle}, "
        f"lr={args.lr}, params={train_p:,}, "
        f"edge_weight={getattr(args, 'edge_weight', 1.0)}"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs("checkpoints_cyclevae", exist_ok=True)

    print(f"\n=== Training ({args.epochs} epochs) ===")
    best_val = float('inf')
    val_stats = {'loss': float('nan'), 'recon': float('nan'), 'cycle': float('nan')}

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        writer.add_scalar("train/loss", train_stats['loss'], epoch)
        writer.add_scalar("train/recon", train_stats['recon'], epoch)
        writer.add_scalar("train/cycle", train_stats['cycle'], epoch)
        writer.add_scalar("train/kl", train_stats['kl'], epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/recon_a", train_stats['recon_a'], epoch)
        writer.add_scalar("train/recon_b", train_stats['recon_b'], epoch)
        writer.add_scalar("train/cycle_a", train_stats['cycle_a'], epoch)
        writer.add_scalar("train/cycle_b", train_stats['cycle_b'], epoch)

        val_after = getattr(args, 'val_after_X_epochs', 1)
        if epoch == 1 or epoch % val_after == 0 or epoch == args.epochs:
            val_stats = validate_epoch(model, val_loader, device)

            writer.add_scalar("val/loss", val_stats['loss'], epoch)
            writer.add_scalar("val/recon", val_stats['recon'], epoch)
            writer.add_scalar("val/cycle", val_stats['cycle'], epoch)
            writer.add_scalar("val/recon_a", val_stats['recon_a'], epoch)
            writer.add_scalar("val/recon_b", val_stats['recon_b'], epoch)
            writer.add_scalar("val/cycle_a", val_stats['cycle_a'], epoch)
            writer.add_scalar("val/cycle_b", val_stats['cycle_b'], epoch)

            print(f"\n  Ep {epoch}: train={train_stats['loss']:.4f} "
                  f"(Ra={train_stats['recon_a']:.4f} Rb={train_stats['recon_b']:.4f} "
                  f"Ca={train_stats['cycle_a']:.4f} Cb={train_stats['cycle_b']:.4f} "
                  f"KL={train_stats['kl']:.4f}), "
                  f"val={val_stats['loss']:.4f} "
                  f"(Ra={val_stats['recon_a']:.4f} Rb={val_stats['recon_b']:.4f} "
                  f"Ca={val_stats['cycle_a']:.4f} Cb={val_stats['cycle_b']:.4f}), "
                  f"lr={lr:.6f}")

            torch.save(model.state_dict(), f"checkpoints_cyclevae/epoch_{epoch}.pt")
            if val_stats['loss'] < best_val:
                best_val = val_stats['loss']
                torch.save(model.state_dict(), "checkpoints_cyclevae/best.pt")
                print(f"  ** Best: {best_val:.4f} **")

    torch.save(model.state_dict(), "checkpoints_cyclevae/final.pt")
    writer.close()
    print(f"\n=== Done. Best val: {best_val:.4f} ===")