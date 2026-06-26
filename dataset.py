"""
dataset.py — PyG Dataset for cortical surface graphs from FreeSurfer output
============================================================================

Reads preprocessed .npz files produced by run_fastsurfer_2.py and builds
PyG Data objects for the Graph Beta-VAE model.

Per subject/hemisphere the pipeline produces:
  {hemi}.morpho.npz   — (163842, 4-5) thickness, curv, sulc, area, [wg_pct]

Shared (fsaverage_common/):
  {hemi}.positions.npy — (163842, 3) vertex coordinates
  {hemi}.edge_index.pt — (2, E) cached mesh connectivity

Patch extraction:
  The full fsaverage hemisphere has 163,842 vertices — too large for
  graph pooling. Each __getitem__ call extracts a random BFS
  patch of `patch_size` nodes (default 10000), creating a contiguous
  subgraph.

  No masking is applied — the VAE reconstructs ALL nodes from the
  latent bottleneck. Anomaly detection uses per-node reconstruction
  error at test time.

Each PyG Data object:
  data.x            (P, F)   cortical features (P = patch_size)
  data.edge_index   (2, E')  local mesh connectivity
  data.edge_attr    (E', 4)  [dist_norm, unit_dx, unit_dy, unit_dz]
  data.pos          (P, 3)   vertex coordinates
  data.atlas_label  (P,)     DKT atlas parcellation label
  data.has_anomaly  bool
"""

import logging
import os
import pathlib
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data, Dataset

# masking import removed — VAE reconstructs all nodes, no masking needed

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

N_FSAVERAGE_VERTICES = 163842
MORPHO_FEATURES = ["thickness", "curv", "sulc", "area"]
HEMIS = ("lh", "rh")
ATLAS_CONFIG = {
    "dkt": {
        "num_regions": 36,
        "per_subject": True,
        "file": "{hemi}.aparc.DKTatlas.fsaverage.annot",
        "dir": "fsaverage_features",
    },
    "destrieux": {
        "num_regions": 76,
        "per_subject": False,
        "file": "{hemi}.aparc.a2009s.annot",
        "dir": "label",
    },
    "hcp": {
        "num_regions": 181,
        "per_subject": False,
        "file": "{hemi}.HCP-MMP1.annot",
        "dir": "label",
    },
}


# ─────────────────────────────────────────────────────────
# Mesh loading (fsaverage — shared across all subjects)
# ─────────────────────────────────────────────────────────

def load_fsaverage_mesh(subjects_dir: str, hemi: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load fsaverage mesh: vertex positions + edge_index from triangular faces.
    Caches edge_index to {hemi}.edge_index.pt for fast reloading.
    """
    common_dir = pathlib.Path(subjects_dir) / "fsaverage_common"
    pos_file = common_dir / f"{hemi}.positions.npy"
    if not pos_file.exists():
        raise FileNotFoundError(f"fsaverage positions not found: {pos_file}")

    positions = torch.from_numpy(np.load(str(pos_file))).float()

    # Cached edge_index
    edge_file = common_dir / f"{hemi}.edge_index.pt"
    if edge_file.exists():
        edge_index = torch.load(str(edge_file), weights_only=True)
        return edge_index, positions

    # Build from fsaverage surface
    fsavg_surf = pathlib.Path(subjects_dir) / "fsaverage" / "surf" / f"{hemi}.white"
    if fsavg_surf.exists():
        edge_index = _edge_index_from_surface(str(fsavg_surf))
    else:
        logging.warning(f"fsaverage surface not found, Delaunay fallback for {hemi}")
        edge_index = _edge_index_delaunay(positions)

    torch.save(edge_index, str(edge_file))
    logging.info(f"Cached edge_index: {edge_file} ({edge_index.shape[1]} edges)")
    return edge_index, positions


def _edge_index_from_surface(surf_path: str) -> torch.Tensor:
    """Build edge_index from FreeSurfer triangle surface file."""
    try:
        import nibabel.freesurfer as fs
        _, faces = fs.read_geometry(surf_path)
    except ImportError:
        with open(surf_path, "rb") as f:
            magic = f.read(3)
            if magic != b"\xff\xff\xfe":
                raise ValueError(f"Not a FreeSurfer surface: {surf_path}")
            while True:
                c = f.read(1)
                if c == b"\n":
                    if f.read(1) == b"\n":
                        break
            n_verts, n_faces = struct.unpack(">ii", f.read(8))
            f.read(n_verts * 3 * 4)
            faces = np.frombuffer(f.read(n_faces * 3 * 4), dtype=">i4").reshape(n_faces, 3)

    edges = set()
    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    edges = list(edges)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def _edge_index_delaunay(positions: torch.Tensor) -> torch.Tensor:
    """Fallback: Delaunay on 2D projection."""
    tri = Delaunay(positions.numpy()[:, :2])
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    edges = list(edges)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


# ─────────────────────────────────────────────────────────
# Edge features
# ─────────────────────────────────────────────────────────

def compute_edge_features(edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Edge attr: [dist_norm, unit_dx, unit_dy, unit_dz]."""
    src, dst = edge_index
    delta = pos[dst].float() - pos[src].float()
    dist = delta.norm(p=2, dim=1, keepdim=True)
    unit = delta / (dist + 1e-8)
    dist_norm = dist / (dist.max() + 1e-8)
    return torch.cat([dist_norm, unit], dim=1)


# ─────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────

def extract_bfs_patch(
    edge_index: torch.Tensor,
    num_nodes: int,
    patch_size: int = 5000,
    center_node: Optional[int] = None,
    adj_cache=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a contiguous subgraph patch via BFS from a random center.

    Returns a local patch with remapped contiguous node indices.
    Edge features are NOT included — recompute from local positions
    so that dist_norm is relative to the patch (not the full hemisphere).

    Args:
        edge_index:  (2, E) full mesh edge index
        num_nodes:   total nodes in the full mesh
        patch_size:  target number of nodes in the patch
        center_node: BFS start (random if None)
        adj_cache:   precomputed scipy sparse adjacency (avoids recomputation)

    Returns:
        patch_nodes:      (P,) long — global indices of selected nodes
        local_edge_index: (2, E_local) — edges remapped to [0, P)
    """
    import random
    from scipy.sparse.csgraph import breadth_first_order

    patch_size = min(patch_size, num_nodes)

    if adj_cache is not None:
        adj = adj_cache
    else:
        from torch_geometric.utils import to_scipy_sparse_matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    if center_node is None:
        center_node = random.randint(0, num_nodes - 1)

    order, _ = breadth_first_order(adj, center_node, directed=False)
    patch_nodes = torch.from_numpy(order[:patch_size]).long()

    # Build node membership set for fast edge filtering
    in_patch = torch.zeros(num_nodes, dtype=torch.bool)
    in_patch[patch_nodes] = True

    # Keep only edges where BOTH endpoints are in the patch
    src, dst = edge_index
    keep = in_patch[src] & in_patch[dst]
    patch_edge_index = edge_index[:, keep]

    # Remap global node indices -> local [0, patch_size)
    mapping = torch.full((num_nodes,), -1, dtype=torch.long)
    mapping[patch_nodes] = torch.arange(len(patch_nodes))
    local_edge_index = mapping[patch_edge_index]

    return patch_nodes, local_edge_index


# ─────────────────────────────────────────────────────────
# Per-subject feature loading
# ─────────────────────────────────────────────────────────

def load_atlas_labels(subjects_dir: str, subject_id: str, hemi: str, atlas: str = "dkt") -> Optional[torch.Tensor]:
    """
    Load atlas parcellation labels on fsaverage.

    For per-subject atlases (dkt): loads from subject's fsaverage_features dir.
    For global atlases (destrieux, hcp): loads from fsaverage/label dir.
    """
    cfg = ATLAS_CONFIG[atlas]
    if cfg["per_subject"]:
        annot_path = pathlib.Path(subjects_dir) / subject_id / cfg["dir"] / cfg["file"].format(hemi=hemi)
    else:
        annot_path = pathlib.Path(subjects_dir) / "fsaverage" / cfg["dir"] / cfg["file"].format(hemi=hemi)

    if not annot_path.exists():
        return None

    import nibabel.freesurfer as fs
    labels, _, _ = fs.read_annot(str(annot_path))
    return torch.from_numpy(np.array(labels, dtype=np.int64)).long()


def load_subject_features(
    subjects_dir: str,
    subject_id: str,
    hemi: str,
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """
    Load morphometric features for one subject/hemisphere.

    Returns (features, feature_names) or None if missing.
    """
    sd = pathlib.Path(subjects_dir) / subject_id

    morpho_file = sd / f"{hemi}.morpho.npz"
    if not morpho_file.exists():
        return None
    morpho = np.load(str(morpho_file))
    data = morpho["data"]
    if data.shape[0] != N_FSAVERAGE_VERTICES:
        return None
    names = list(morpho.get("feature_names", MORPHO_FEATURES))
    features = data.astype(np.float32)
    return torch.from_numpy(features), names


# ─────────────────────────────────────────────────────────
# Main Dataset
# ─────────────────────────────────────────────────────────

class CorticalSurfaceDataset(Dataset):
    """
    PyG Dataset of real cortical surface graphs (morpho features only).

    Args:
        data_dicts:       List of dicts from build_dataset
        subjects_dir:     Path to FastSurfer subjects directory
        hemispheres:      Which hemispheres ('lh', 'rh')
        patch_size:       BFS patch size
        normalize:        Z-score normalize features
        mesh_cache:       Shared mesh cache
    """

    def __init__(
        self,
        data_dicts: List[dict],
        subjects_dir: str,
        hemispheres: List[str] = None,
        patch_size: int = 5000,
        mask_ratio_range: Tuple[float, float] = (0.15, 0.40),
        normalize: bool = True,
        mesh_cache: Optional[dict] = None,
        norm_stats: Optional[dict] = None,
        atlas: str = "dkt",
    ):
        super().__init__(root=None, transform=None, pre_transform=None)

        self.atlas = atlas
        self.subjects_dir = subjects_dir
        self.hemispheres = hemispheres or list(HEMIS)
        self.patch_size = patch_size
        self.mask_ratio_range = mask_ratio_range
        self.normalize = normalize

        # Shared mesh (same for all subjects in fsaverage space)
        if mesh_cache is not None:
            self.mesh_cache = mesh_cache
        else:
            self.mesh_cache = {}
            for hemi in self.hemispheres:
                edge_index, positions = load_fsaverage_mesh(subjects_dir, hemi)
                edge_attr = compute_edge_features(edge_index, positions)
                self.mesh_cache[hemi] = {
                    "edge_index": edge_index,
                    "positions": positions,
                    "edge_attr": edge_attr,
                }

        # Pre-compute scipy adjacency for fast BFS patch extraction
        if self.patch_size > 0 and self.patch_size < N_FSAVERAGE_VERTICES:
            from torch_geometric.utils import to_scipy_sparse_matrix
            for hemi in self.hemispheres:
                mesh = self.mesh_cache[hemi]
                if "adj_scipy" not in mesh:
                    mesh["adj_scipy"] = to_scipy_sparse_matrix(
                        mesh["edge_index"], num_nodes=N_FSAVERAGE_VERTICES
                    )

        # Load features for each subject × hemisphere
        self.samples = []
        self.feature_names = None
        self._num_node_features = 0
        n_skipped = 0

        for dd in data_dicts:
            sid = dd["study_uid"]
            has_anomaly = len(dd.get("seg_masks", [])) > 0

            # Per-dict subjects_dir support: each loaded dict may carry its own
            # subjects_dir (combined training pools more than one cohort).
            # Falls back to the class-level subjects_dir.
            sd_per_dict = dd.get("subjects_dir", subjects_dir)

            for hemi in self.hemispheres:
                result = load_subject_features(sd_per_dict, sid, hemi)
                if result is None:
                    n_skipped += 1
                    continue

                features, feat_names = result
                if self.feature_names is None:
                    self.feature_names = feat_names
                    self._num_node_features = features.shape[1]

                atlas_labels = load_atlas_labels(sd_per_dict, sid, hemi, atlas=self.atlas)
                if atlas_labels is None:
                    logging.warning(f"No atlas labels for {sid} {hemi}, skipping")
                    n_skipped += 1
                    continue

                self.samples.append({
                    "features": features,
                    "atlas_labels": atlas_labels,
                    "subject_id": sid,
                    "hemi": hemi,
                    "has_anomaly": has_anomaly,
                    "befund": dd.get("study_befund", "unknown"),
                    "city": dd.get("city", "unknown"),
                    "subjects_dir": sd_per_dict,
                    "seg_masks": dd.get("seg_masks", []),
                })

        if n_skipped > 0:
            logging.info(f"Skipped {n_skipped} subject×hemi (missing morpho or atlas)")
        logging.info(f"Loaded {len(self.samples)} graphs from {len(data_dicts)} subjects")

        # Normalization
        if norm_stats is not None:
            self.feat_mean = norm_stats["mean"]
            self.feat_std = norm_stats["std"]
        elif self.normalize and self.samples:
            self._compute_norm_stats()

    def _compute_norm_stats(self):
        """Per-feature z-score stats, ignoring zeros (missing data)."""
        all_feat = torch.stack([s["features"] for s in self.samples])
        self.feat_mean = torch.zeros(self._num_node_features)
        self.feat_std = torch.ones(self._num_node_features)
        for f in range(self._num_node_features):
            vals = all_feat[:, :, f]
            nonzero = vals[vals != 0]
            if len(nonzero) > 100:
                self.feat_mean[f] = nonzero.mean()
                self.feat_std[f] = nonzero.std().clamp(min=1e-6)

    def get_norm_stats(self) -> dict:
        """Return normalization stats for sharing with val/test datasets."""
        return {
            "mean": getattr(self, "feat_mean", None),
            "std": getattr(self, "feat_std", None),
        }

    def len(self):
        return len(self.samples)

    def get(self, idx):
        item = self.samples[idx]
        hemi = item["hemi"]
        mesh = self.mesh_cache[hemi]

        features = item["features"].clone()
        if self.normalize and hasattr(self, "feat_mean"):
            features = (features - self.feat_mean) / self.feat_std

        full_edge_index = mesh["edge_index"]
        full_positions = mesh["positions"]

        # ── Step 1: Extract a local BFS patch ──────────────────────
        # Random center each access → different patch every epoch.
        # patch_size=0 means use full hemisphere (no patching).
        patch_nodes, local_edge_index = extract_bfs_patch(
            full_edge_index, N_FSAVERAGE_VERTICES,
            patch_size=self.patch_size,
            adj_cache=mesh.get("adj_scipy"),
        )
        n_local = len(patch_nodes)

        # Subset features, positions, atlas labels to patch
        local_features = features[patch_nodes]
        local_pos = full_positions[patch_nodes]
        num_regions = ATLAS_CONFIG[self.atlas]["num_regions"]
        local_atlas = item["atlas_labels"][patch_nodes].clamp(0, num_regions - 1)

        # Recompute edge_attr from local positions so dist_norm
        # is relative to the patch, not the full hemisphere
        local_edge_attr = compute_edge_features(local_edge_index, local_pos)

        # ── Step 2: Build Data object (no masking for VAE) ────────
        # VAE reconstructs all nodes from the latent bottleneck.
        # No mask needed — anomaly score = per-node reconstruction error.

        return Data(
            x=local_features,
            edge_index=local_edge_index,
            edge_attr=local_edge_attr,
            pos=local_pos,
            atlas_label=local_atlas,
            has_anomaly=item["has_anomaly"],
            num_nodes=n_local,
        )

    @property
    def num_node_features(self):
        return self._num_node_features