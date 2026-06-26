"""
cycle_vae.py — Cycle-consistent Graph VAE for cortical anomaly detection
==========================================================================

Two VAE paths with cycle consistency:
  Path A: morpho → z_a → FLAIR_pred    (morpho-to-modality)
  Path B: FLAIR  → z_b → morpho_pred   (modality-to-morpho)

Cycle consistency:
  morpho → FLAIR_pred → morpho_recon   (should match original morpho)
  FLAIR  → morpho_pred → FLAIR_recon   (should match original FLAIR)

Total loss = recon_A + recon_B + λ_cycle * (cycle_A + cycle_B) + β * (KL_A + KL_B)

Anomaly score at test time:
  score = recon_error_A + recon_error_B + cycle_error_A + cycle_error_B
  (A lesion disrupts BOTH directions of the morpho↔FLAIR mapping)

Architecture reuses GatedGCNBlock, GraphVAEEncoder, GraphVAEDecoder from
graph_beta_vae.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SAGPooling, global_mean_pool


from gatedgcn_layer import GatedGCNLayer
from mni_pos_encoder import CorticalPEEncoder, EdgeSpatialEncoder


# ─────────────────────────────────────────────────────────────
#  Reused building blocks (identical to graph_beta_vae.py)
# ─────────────────────────────────────────────────────────────

class GatedGCNBlock(nn.Module):
    def __init__(self, dim_h, num_layers=2, dropout=0.0, act='relu'):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedGCNLayer(dim_h, dim_h, dropout=dropout, residual=True, act=act)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.layers:
            b = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            b = layer(b)
            x, edge_attr = b.x, b.edge_attr
        return x, edge_attr


class GraphVAEEncoder(nn.Module):
    def __init__(self, in_dim, dim_h, latent_dim, num_pool_levels=3,
                 pool_ratio=0.5, gnn_layers_per_level=2, dropout=0.0, act='relu', edge_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.num_pool_levels = num_pool_levels
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, dim_h), nn.BatchNorm1d(dim_h), nn.ReLU())
        self.edge_encoder = EdgeSpatialEncoder(in_edge_dim=4, dim_h=dim_h)
        self.gnn_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(num_pool_levels):
            self.gnn_blocks.append(GatedGCNBlock(dim_h, gnn_layers_per_level, dropout, act))
            self.pools.append(SAGPooling(dim_h, ratio=pool_ratio))
        self.final_gnn = GatedGCNBlock(dim_h, num_layers=1, dropout=dropout, act=act)
        self.fc_mu = nn.Linear(dim_h, latent_dim)
        self.fc_logvar = nn.Linear(dim_h, latent_dim)

    def forward(self, x, edge_index, edge_attr_raw, pos, batch):
        h = self.input_proj(x)
        # edge_attr_raw = torch.zeros_like(edge_attr_raw) # ablation without edges
        edge_attr = self.edge_encoder(edge_attr_raw)
        # edge_attr = self.edge_encoder(self.edge_weight * edge_attr_raw)
        pool_records = []
        for i in range(self.num_pool_levels):
            h, edge_attr = self.gnn_blocks[i](h, edge_index, edge_attr, batch)
            record = {'x_pre': h, 'edge_index_pre': edge_index,
                      'edge_attr_pre': edge_attr, 'batch_pre': batch,
                      'num_nodes_pre': h.size(0)}
            pool_out = self.pools[i](h, edge_index, edge_attr=edge_attr, batch=batch)
            h, edge_index, edge_attr, batch, perm, score = pool_out
            record.update({'perm': perm, 'score': score,
                           'num_nodes_post': h.size(0), 'batch_post': batch})
            pool_records.append(record)
        h, edge_attr = self.final_gnn(h, edge_index, edge_attr, batch)
        h_graph = global_mean_pool(h, batch)
        return self.fc_mu(h_graph), self.fc_logvar(h_graph), pool_records


class GraphVAEDecoder(nn.Module):
    def __init__(self, dim_h, latent_dim, out_dim, num_pool_levels=3,
                 gnn_layers_per_level=2, dropout=0.0, act='relu'):
        super().__init__()
        self.num_pool_levels = num_pool_levels
        self.dim_h = dim_h
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_h))
        self.fill_token = nn.Parameter(torch.zeros(1, dim_h))
        nn.init.normal_(self.fill_token, std=0.02)
        self.gnn_blocks = nn.ModuleList()
        for _ in range(num_pool_levels):
            self.gnn_blocks.append(GatedGCNBlock(dim_h, gnn_layers_per_level, dropout, act))
        ACT_CLS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'elu': nn.ELU}[act]
        self.output_head = nn.Sequential(
            nn.Linear(dim_h, dim_h // 2), ACT_CLS(),
            nn.Linear(dim_h // 2, dim_h // 4), ACT_CLS(),
            nn.Linear(dim_h // 4, out_dim))

    def forward(self, z, pool_records, batch_sizes):
        h_graph = self.latent_proj(z)
        last_record = pool_records[-1]
        h = h_graph[last_record['batch_post']]
        for i in range(self.num_pool_levels - 1, -1, -1):
            rec = pool_records[i]
            h = h.float()
            edge_attr_pre = rec['edge_attr_pre'].float()
            h_unpooled = self.fill_token.expand(rec['num_nodes_pre'], -1).clone()
            h_unpooled[rec['perm']] = h
            h, edge_attr_pre = self.gnn_blocks[self.num_pool_levels - 1 - i](
                h_unpooled, rec['edge_index_pre'], edge_attr_pre, rec['batch_pre'])
        return self.output_head(h)


# ─────────────────────────────────────────────────────────────
#  Single VAE path (morpho→FLAIR or FLAIR→morpho)
# ─────────────────────────────────────────────────────────────

class VAEPath(nn.Module):
    """One direction of the cycle: encoder(source) → z → decoder(target)."""

    def __init__(self, in_feat_dim, out_feat_dim, dim_h, dim_pe, latent_dim,
                 num_pool_levels, pool_ratio, gnn_layers_per_level, dropout, act,
                 num_atlas_regions=36, dim_atlas=16, edge_weight=1.0):
        super().__init__()
        self.pe_encoder = CorticalPEEncoder(
            dim_pe=dim_pe, dim_mni=32, dim_atlas=dim_atlas, num_regions=num_atlas_regions, sigma=0.05)

        self.encoder = GraphVAEEncoder(
            in_dim=in_feat_dim + dim_pe, dim_h=dim_h, latent_dim=latent_dim,
            num_pool_levels=num_pool_levels, pool_ratio=pool_ratio,
            gnn_layers_per_level=gnn_layers_per_level, dropout=dropout, act=act, edge_weight=edge_weight)

        self.decoder = GraphVAEDecoder(
            dim_h=dim_h, latent_dim=latent_dim, out_dim=out_feat_dim,
            num_pool_levels=num_pool_levels,
            gnn_layers_per_level=gnn_layers_per_level, dropout=dropout, act=act)

    def encode(self, x_source, edge_index, edge_attr, pos, atlas_label, batch):
        pe = self.pe_encoder(pos, atlas_label)
        # pe = torch.zeros_like(pe)  # ABLATION A5c: no positional encoding
        x_in = torch.cat([x_source, pe], dim=-1)
        mu, logvar, pool_records = self.encoder(x_in, edge_index, edge_attr, pos, batch)
        return mu, logvar, pool_records

    def decode(self, z, pool_records, batch_size):
        return self.decoder(z, pool_records, batch_size)

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x_source, edge_index, edge_attr, pos, atlas_label, batch):
        mu, logvar, pool_records = self.encode(
            x_source, edge_index, edge_attr, pos, atlas_label, batch)
        z = self.reparameterize(mu, logvar, self.training)
        recon = self.decode(z, pool_records, mu.size(0))
        return recon, mu, logvar, pool_records


# ─────────────────────────────────────────────────────────────
#  CycleVAE: two paths + cycle consistency
# ─────────────────────────────────────────────────────────────

class GraphCycleVAE(nn.Module):
    """
    Cycle-consistent dual Graph VAE.

    Supports flexible feature splits:
      split_mode='morpho_mod':  Path A: morpho(5) ↔ modality(7)   [default]
      split_mode='morpho_only': Path A: geometry(curv,sulc) ↔ structure(thick,area,wg_pct)
      split_mode='custom':      Path A: indices_a ↔ indices_b     [user-defined]

    Cycle A: feat_a → feat_b_pred → feat_a_recon
    Cycle B: feat_b → feat_a_pred → feat_b_recon

    Args:
        num_features:   total number of input features
        split_mode:     'morpho_mod' | 'morpho_only' | 'custom'
        num_morpho:     number of morphometric features (used for morpho_mod)
        indices_a:      list of feature indices for path A input (custom mode)
        indices_b:      list of feature indices for path B input (custom mode)
        dim_h:          GNN hidden dimension
        dim_pe:         positional encoding dimension
        latent_dim:     VAE latent dimension
        beta:           KL weight
        lambda_cycle:   cycle consistency weight
    """

    def __init__(
        self,
        num_features: int = 12,
        split_mode: str = 'morpho_mod',
        num_morpho: int = 5,
        indices_a: list = None,
        indices_b: list = None,
        dim_h: int = 128,
        dim_pe: int = 32,
        latent_dim: int = 64,
        num_pool_levels: int = 3,
        pool_ratio: float = 0.5,
        gnn_layers_per_level: int = 2,
        beta: float = 1.0,
        lambda_cycle: float = 1.0,
        dropout: float = 0.0,
        act: str = 'relu',
        num_atlas_regions: int = 36,
        dim_atlas: int = 16,
        edge_weight: float = 1.0,
        # Legacy compat — ignored if split_mode is used
        num_morpho_legacy: int = None,
        num_mod: int = None,
    ):
        super().__init__()
        self.beta = beta
        self.lambda_cycle = lambda_cycle
        self.split_mode = split_mode

        # Determine feature index splits
        if split_mode == 'morpho_mod':
            # morpho features [0..num_morpho) ↔ modality features [num_morpho..num_features)
            self.indices_a = list(range(num_morpho))
            self.indices_b = list(range(num_morpho, num_features))
        elif split_mode == 'morpho_only':
            # geometry ↔ structure split within morpho features
            # Feature order: thickness(0), curv(1), sulc(2), area(3), [wg_pct(4)]
            self.indices_a = [1, 2]          # geometry: curvature, sulcal depth
            if num_features >= 5:
                self.indices_b = [0, 3, 4]   # structure: thickness, area, wg_pct
            else:
                self.indices_b = [0, 3]      # structure: thickness, area (no wg_pct)
        elif split_mode == 'custom':
            assert indices_a is not None and indices_b is not None
            self.indices_a = indices_a
            self.indices_b = indices_b
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        n_a = len(self.indices_a)
        n_b = len(self.indices_b)

        # Path A: feat_a → feat_b
        self.path_a = VAEPath(
            in_feat_dim=n_a, out_feat_dim=n_b,
            dim_h=dim_h, dim_pe=dim_pe, latent_dim=latent_dim,
            num_pool_levels=num_pool_levels, pool_ratio=pool_ratio,
            gnn_layers_per_level=gnn_layers_per_level, dropout=dropout, act=act,
            num_atlas_regions=num_atlas_regions, dim_atlas=dim_atlas, edge_weight=edge_weight)

        # Path B: feat_b → feat_a
        self.path_b = VAEPath(
            in_feat_dim=n_b, out_feat_dim=n_a,
            dim_h=dim_h, dim_pe=dim_pe, latent_dim=latent_dim,
            num_pool_levels=num_pool_levels, pool_ratio=pool_ratio,
            gnn_layers_per_level=gnn_layers_per_level, dropout=dropout, act=act,
            num_atlas_regions=num_atlas_regions, dim_atlas=dim_atlas, edge_weight=edge_weight)

    def _split(self, x):
        """Split input features into group A and group B."""
        return x[:, self.indices_a], x[:, self.indices_b]

    def forward(self, batch):
        """
        Full forward: both paths + cycle.
        Returns dict with all intermediate results for loss computation.
        """
        feat_a, feat_b = self._split(batch.x)
        ei, ea, pos, al, b = (batch.edge_index, batch.edge_attr,
                               batch.pos, batch.atlas_label, batch.batch)

        # Path A: feat_a → feat_b_pred
        feat_b_pred, mu_a, logvar_a, records_a = self.path_a(feat_a, ei, ea, pos, al, b)

        # Path B: feat_b → feat_a_pred
        feat_a_pred, mu_b, logvar_b, records_b = self.path_b(feat_b, ei, ea, pos, al, b)

        # Cycle A: feat_b_pred → feat_a_recon (use path B on predicted feat_b)
        feat_a_cycle, mu_cycle_a, logvar_cycle_a, _ = self.path_b(
            feat_b_pred, ei, ea, pos, al, b) # feat_b_pred.detach()

        # Cycle B: feat_a_pred → feat_b_recon (use path A on predicted feat_a)
        feat_b_cycle, mu_cycle_b, logvar_cycle_b, _ = self.path_a(
            feat_a_pred, ei, ea, pos, al, b) # feat_a_pred.detach()

        return {
            'feat_b_pred': feat_b_pred, 'feat_a_pred': feat_a_pred,
            'feat_a_cycle': feat_a_cycle, 'feat_b_cycle': feat_b_cycle,
            'mu_a': mu_a, 'logvar_a': logvar_a,
            'mu_b': mu_b, 'logvar_b': logvar_b,
            'feat_a': feat_a, 'feat_b': feat_b,
        }

    def compute_loss(self, out, kld_weight=1.0):
        """
        Combined loss:
          recon_a:  MSE(feat_b_pred, feat_b_true)
          recon_b:  MSE(feat_a_pred, feat_a_true)
          cycle_a:  MSE(feat_a_cycle, feat_a_true)
          cycle_b:  MSE(feat_b_cycle, feat_b_true)
          kl_a + kl_b: KL divergence for both paths
        """
        recon_a = F.mse_loss(out['feat_b_pred'], out['feat_b'])
        recon_b = F.mse_loss(out['feat_a_pred'], out['feat_a'])
        cycle_a = F.mse_loss(out['feat_a_cycle'], out['feat_a'])
        cycle_b = F.mse_loss(out['feat_b_cycle'], out['feat_b'])

        kl_a = torch.mean(-0.5 * torch.sum(
            1 + out['logvar_a'] - out['mu_a'].pow(2) - out['logvar_a'].exp(), dim=1))
        kl_b = torch.mean(-0.5 * torch.sum(
            1 + out['logvar_b'] - out['mu_b'].pow(2) - out['logvar_b'].exp(), dim=1))

        recon_loss = recon_a + recon_b
        cycle_loss = cycle_a + cycle_b
        kl_loss = kl_a + kl_b

        total = recon_loss + self.lambda_cycle * cycle_loss + self.beta * kld_weight * kl_loss

        return {
            'loss': total,
            'recon_loss': recon_loss,
            'recon_a': recon_a, 'recon_b': recon_b,
            'cycle_loss': cycle_loss,
            'cycle_a': cycle_a, 'cycle_b': cycle_b,
            'kl_loss': kl_loss,
        }

    @torch.no_grad()
    def compute_anomaly_score(self, batch):
        """
        Per-node anomaly score combining both directions + cycle.
        score_i = MSE_a(i) + MSE_b(i) + MSE_cycle_a(i) + MSE_cycle_b(i)
        Returns (N,) scores.
        """
        self.eval()
        out = self.forward(batch)

        err_a = ((out['feat_b_pred'] - out['feat_b']) ** 2).mean(dim=-1)
        err_b = ((out['feat_a_pred'] - out['feat_a']) ** 2).mean(dim=-1)
        err_cycle_a = ((out['feat_a_cycle'] - out['feat_a']) ** 2).mean(dim=-1)
        err_cycle_b = ((out['feat_b_cycle'] - out['feat_b']) ** 2).mean(dim=-1)

        return err_a + err_b + err_cycle_a + err_cycle_b

    @torch.no_grad()
    def compute_anomaly_score_per_feature(self, batch):
        """(N, num_features) per-vertex error, one column per ORIGINAL feature.
        Column f = global feature f. Squared error → MAGNITUDE only, no direction."""
        self.eval()
        out = self.forward(batch)
        err_b_direct = (out['feat_b_pred'] - out['feat_b']) ** 2
        err_a_direct = (out['feat_a_pred'] - out['feat_a']) ** 2
        err_a_cycle = (out['feat_a_cycle'] - out['feat_a']) ** 2
        err_b_cycle = (out['feat_b_cycle'] - out['feat_b']) ** 2
        struct_err = err_b_direct + err_b_cycle  # cols align to self.indices_b
        geom_err = err_a_direct + err_a_cycle  # cols align to self.indices_a
        per_feat = torch.zeros(batch.x.size(0), batch.x.size(1),
                               device=batch.x.device, dtype=struct_err.dtype)
        per_feat[:, self.indices_b] = struct_err
        per_feat[:, self.indices_a] = geom_err
        return per_feat


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable