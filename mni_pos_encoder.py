"""
mni_pos_encoder.py — Anatomically-aware positional encoders for cortical graphs
================================================================================

Node encoders:
  MNIPositionalEncoder  — RFF encoding of 3D MNI coordinates
  AtlasTokenEncoder     — learnable embedding per atlas parcellation region
  CorticalPEEncoder     — combines both -> drop-in for LapPENodeEncoder

Edge encoder:
  EdgeSpatialEncoder    — encodes (E, 4) = [dist_norm, unit_dx, unit_dy, unit_dz]
                          captures geodesic length AND 3D orientation in MNI space
"""

import torch
import torch.nn as nn


class MNIPositionalEncoder(nn.Module):
    """
    Random Fourier Feature encoder for 3D MNI coordinates.

    Maps (N, 3) -> (N, dim_pe) using sinusoidal basis functions.
    Encodes absolute anatomical location: frontal vs parietal, left vs right.

    Unlike LapPE, this is patient-specific: two nodes at the same MNI
    position in different subjects get the same encoding, which is correct
    because they represent the same anatomy.

    Args:
        dim_pe:    Output dimension (must be even for sin+cos pairs)
        sigma:     RFF bandwidth. For raw MNI mm coords (+-80mm): use ~0.05.
    """

    def __init__(self, dim_pe: int = 32, sigma: float = 0.05):
        super().__init__()
        assert dim_pe % 2 == 0, "dim_pe must be even (sin+cos pairs)"
        half   = dim_pe // 2
        B_init = torch.randn(3, half) * sigma
        self.register_buffer("B", B_init)
        self.proj = nn.Sequential(
            nn.Linear(dim_pe, dim_pe),
            nn.ReLU(),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (N, 3) MNI coordinates in mm
        Returns:
            pe:  (N, dim_pe)
        """
        x  = pos.float() @ self.B
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(pe)


class AtlasTokenEncoder(nn.Module):
    """
    Learnable embedding per atlas parcellation region.

    In real FreeSurfer data: labels from aparc.annot (Desikan/Destrieux atlas).
    Encodes semantic anatomical identity: fusiform vs motor cortex, etc.

    Args:
        num_regions: Number of atlas parcellation regions
        dim_atlas:   Embedding dimension per region
    """

    def __init__(self, num_regions: int = 32, dim_atlas: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_regions, dim_atlas)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, atlas_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atlas_label: (N,) long tensor in [0, num_regions)
        Returns:
            emb: (N, dim_atlas)
        """
        return self.embedding(atlas_label)


class CorticalPEEncoder(nn.Module):
    """
    Combined positional encoder: MNI coordinates + atlas parcellation.
    Drop-in replacement for LapPENodeEncoder.

    Why better than LapPE for FreeSurfer meshes:
      - LapPE encodes graph topology, which is nearly identical across all
        subjects (all derived from the same icosphere template via spherical
        registration). Eigenvectors also suffer from sign + basis ambiguity.
      - MNI-RFF encodes absolute anatomical location, which IS patient-specific
        and anatomically meaningful.
      - Atlas tokens encode semantic region identity.

    Output: (N, dim_pe) — same interface as LapPENodeEncoder.

    Args:
        dim_pe:         Output PE dimension
        dim_mni:        Internal MNI RFF dimension
        dim_atlas:      Internal atlas embedding dimension
        num_regions:    Number of atlas parcellation regions
        sigma:          RFF bandwidth for MNI encoder
    """

    def __init__(
        self,
        dim_pe: int = 32,
        dim_mni: int = 32,
        dim_atlas: int = 16,
        num_regions: int = 32,
        sigma: float = 0.05,
    ):
        super().__init__()
        self.dim_pe        = dim_pe
        self.mni_encoder   = MNIPositionalEncoder(dim_pe=dim_mni, sigma=sigma)
        self.atlas_encoder = AtlasTokenEncoder(num_regions=num_regions,
                                               dim_atlas=dim_atlas)
        self.fusion = nn.Sequential(
            nn.Linear(dim_mni + dim_atlas, dim_pe),
            nn.ReLU(),
            nn.Linear(dim_pe, dim_pe),
        )

    def forward(self, pos: torch.Tensor, atlas_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos:         (N, 3) MNI coordinates
            atlas_label: (N,)   atlas labels (long)
        Returns:
            pe:          (N, dim_pe)
        """
        mni_pe    = self.mni_encoder(pos)
        atlas_emb = self.atlas_encoder(atlas_label)
        return self.fusion(torch.cat([mni_pe, atlas_emb], dim=-1))


class EdgeSpatialEncoder(nn.Module):
    """
    Encode full 3D edge spatial features to dim_h.

    Input edge_attr: (E, 4) = [dist_norm, unit_dx, unit_dy, unit_dz]
      - dist_norm:  normalized geodesic length (scalar in [0, 1])
      - unit_dx/dy/dz: unit direction vector in MNI space (3D orientation)

    Why (E,4) instead of scalar distance:
      - Distance alone tells the MPNN how far neighbors are.
      - Direction tells WHERE the neighbor is: sulcal wall vs gyral crown,
        anterior-posterior gradient, radial vs tangential surface direction.
      - GatedGCN gate g_ij = sigma(W[h_i || h_j || e_ij]) can learn to
        weight messages by cortical direction — relevant for FCD boundaries.

    Not rotation-equivariant by design: the FreeSurfer mesh is registered
    to MNI space, so hemisphere and A-P directions are anatomically fixed.

    Args:
        in_edge_dim: Input edge feature dimension (4 for spatial encoder)
        dim_h:       Output dimension (must match GPS hidden dim)
    """

    def __init__(self, in_edge_dim: int = 4, dim_h: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_edge_dim, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: (E, 4) [dist_norm, unit_dx, unit_dy, unit_dz]
        Returns:
            edge_emb:  (E, dim_h)
        """
        return self.encoder(edge_attr.float())