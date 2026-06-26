"""
Laplacian Positional Encoding — adapted from graphgps/encoder/laplace_pos_encoder.py
and graphgps/transform/posenc_stats.py

Changes from original:
  - Removed GraphGym config dependency (explicit constructor args)
  - Removed registry decorators
  - Removed expand_x / concat logic (handled externally)
  - ALL processing logic (DeepSet, sign flip, sum pool, post MLP) is IDENTICAL
  - Eigendecomposition functions copied from posenc_stats.py

Original: https://github.com/rampasek/GraphGPS
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


# ═══════════════════════════════════════════════════════
#  Eigendecomposition — from graphgps/transform/posenc_stats.py
# ═══════════════════════════════════════════════════════

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    Copied IDENTICALLY from posenc_stats.py.
    """
    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
    elif normalization == "wavelength":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom * 2 / np.pi
    elif normalization == "wavelength-asin":
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = eigval_denom
    elif normalization == "wavelength-soft":
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom
    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Copied IDENTICALLY from posenc_stats.py.

    Args:
        evals, evects: Precomputed eigen-decomposition (numpy)
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def compute_posenc_stats(edge_index, num_nodes, is_undirected=True,
                         max_freqs=16, eigvec_norm='L2',
                         laplacian_norm_type=None):
    """Precompute LapPE for a single graph.

    Simplified version of posenc_stats.compute_posenc_stats() — only LapPE.

    Args:
        edge_index: (2, E) tensor
        num_nodes: int
        is_undirected: True for cortical mesh
        max_freqs: Number of eigenvectors to keep
        eigvec_norm: Eigenvector normalization type
        laplacian_norm_type: None or 'sym' or 'rw'

    Returns:
        EigVals: (num_nodes, max_freqs, 1)
        EigVecs: (num_nodes, max_freqs)
    """
    # Eigen-decomposition with numpy (same as original)
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=laplacian_norm_type,
                       num_nodes=num_nodes)
    )
    evals, evects = np.linalg.eigh(L.toarray())

    EigVals, EigVecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=max_freqs,
        eigvec_norm=eigvec_norm)

    return EigVals, EigVecs


# ═══════════════════════════════════════════════════════
#  LapPE Encoder — from graphgps/encoder/laplace_pos_encoder.py
# ═══════════════════════════════════════════════════════

class LapPENodeEncoder(nn.Module):
    """Laplace Positional Embedding node encoder.

    Adapted from graphgps/encoder/laplace_pos_encoder.py.
    Changes:
      - Constructor takes explicit args instead of GraphGym cfg
      - Removed expand_x / concat logic (handled by caller)
      - ALL processing (sign flip, DeepSet/Transformer, sum pool, post MLP) IDENTICAL

    Args:
        dim_pe: Output dimensionality of the PE
        max_freqs: Number of eigenvectors (matches precomputation)
        n_layers: Number of layers in PE encoder
        model_type: 'DeepSet' or 'Transformer'
        post_n_layers: Number of MLP layers after sum pooling
        n_heads: Attention heads (only for Transformer model_type)
        raw_norm_type: Normalization of raw PE ('batchnorm' or 'none')
    """

    def __init__(self, dim_pe=16, max_freqs=16, n_layers=2,
                 model_type='DeepSet', post_n_layers=2, n_heads=1,
                 raw_norm_type='none'):
        super().__init__()
        self.model_type = model_type

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if raw_norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # Same as original

        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        else:
            # DeepSet model for LapPE — IDENTICAL to original
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling — IDENTICAL to original
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)

    def forward(self, EigVecs, EigVals):
        """
        Process Laplacian PE — IDENTICAL logic to original forward().

        Args:
            EigVecs: (N, max_freqs) eigenvector values per node
            EigVals: (N, max_freqs, 1) eigenvalues repeated per node

        Returns:
            pos_enc: (N, dim_pe) positional encoding per node
        """
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2)  # (N, max_freqs, 2)
        empty_mask = torch.isnan(pos_enc)  # (N, max_freqs, 2)

        pos_enc[empty_mask] = 0  # (N, max_freqs, 2)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (N, max_freqs, dim_pe)

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (N, dim_pe)

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (N, dim_pe)

        return pos_enc
