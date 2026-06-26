"""
validation.py — Validation for Graph Cycle-VAE
================================================

Adapted from the GraphBetaVAE validation module.
Only change: _load_model instantiates GraphCycleVAE.
All scoring uses model.compute_anomaly_score() which returns
combined cross-modal + cycle consistency error per vertex.

Three entry points:
  run_healthy_baseline(args, dm)
  run_detection(args, dm)
  run_classification(args, dm)

Usage:
  python run_cyclevae.py --mode baseline
  python run_cyclevae.py --mode detect
  python run_cyclevae.py --mode classify
"""

import json
import logging
import pathlib
from typing import List, Optional
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
from tqdm import tqdm

from dataset import (
    load_fsaverage_mesh, compute_edge_features,
    N_FSAVERAGE_VERTICES,
)
from cycle_vae import GraphCycleVAE, count_params
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ═══════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════

def _load_model(args, device):
    model = GraphCycleVAE(
        num_features=args.num_features,
        split_mode=getattr(args, 'split_mode', 'morpho_mod'),
        num_morpho=args.num_morpho,
        indices_a=getattr(args, 'indices_a', None),
        indices_b=getattr(args, 'indices_b', None),
        dim_h=args.dim_h, dim_pe=getattr(args, 'dim_pe', 32),
        latent_dim=args.latent_dim,
        num_pool_levels=args.num_pool_levels, pool_ratio=args.pool_ratio,
        gnn_layers_per_level=args.gnn_layers_per_level,
        beta=args.beta, lambda_cycle=getattr(args, 'lambda_cycle', 1.0),
        dropout=0.0, act='relu',
        num_atlas_regions=args.num_atlas_regions,
        dim_atlas=args.dim_atlas,
        edge_weight=getattr(args, 'edge_weight', 1.0),
    ).to(device)
    ckpt = torch.load(args.val_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    _, tp = count_params(model)
    logging.info(f"  Model: {tp:,} params, ckpt={args.val_checkpoint}")
    return model


def _load_baseline(args):
    bp = pathlib.Path(args.val_output_dir) / "healthy_baseline.json"
    if not bp.exists():
        raise FileNotFoundError(
            f"No baseline found: {bp}\n"
            f"Run: python run_graphvae.py --mode baseline first"
        )
    with open(bp) as f:
        baseline = json.load(f)

    # Optional per-vertex baseline (one .npz per hemi inside the file).
    # When present, the z-score code uses per-vertex median as the location
    # estimate instead of the per-region mean — captures within-region
    # heterogeneity of the model's healthy reconstruction error.
    vp = pathlib.Path(args.val_output_dir) / "healthy_baseline_vertex.npz"
    if vp.exists():
        data = np.load(str(vp))
        vbase = {}
        for hemi in ("lh", "rh"):
            key = f"{hemi}_median"
            if key in data.files:
                vbase[hemi] = {"median": data[key]}
        baseline["vertex_baselines"] = vbase
        logging.info(
            f"  Loaded per-vertex baseline: hemispheres={list(vbase.keys())}"
        )
    return baseline


def _prepare_meshes(dm):
    from torch_geometric.utils import to_scipy_sparse_matrix
    for hemi in dm.hemispheres:
        mesh = dm.mesh_cache[hemi]
        if "adj_scipy" not in mesh:
            mesh["adj_scipy"] = to_scipy_sparse_matrix(
                mesh["edge_index"], num_nodes=N_FSAVERAGE_VERTICES
            )


def _load_faces(dm, subjects_dir):
    fc = {}
    for hemi in dm.hemispheres:
        fsp = pathlib.Path(subjects_dir) / "fsaverage" / "surf" / f"{hemi}.white"
        if fsp.exists():
            try:
                import nibabel.freesurfer as fs
                _, faces = fs.read_geometry(str(fsp))
                fc[hemi] = faces
            except Exception:
                fc[hemi] = None
        else:
            fc[hemi] = None
    return fc


# ── Coordinate mapping ──

# ═══════════════════════════════════════════════════════════════
#  MNI-space seg mask evaluation (replaces surface projection)
# ═══════════════════════════════════════════════════════════════

def _read_talairach_xfm(subjects_dir, subject_id):
    """Read FreeSurfer talairach.xfm → 4x4 affine (scanner RAS → MNI305)."""
    xfm_path = pathlib.Path(subjects_dir) / subject_id / "mri" / "transforms" / "talairach.xfm"
    if not xfm_path.exists():
        return None
    rows = []
    reading = False
    with open(xfm_path) as f:
        for line in f:
            if "Linear_Transform" in line:
                reading = True
                continue
            if reading:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                vals = [float(v) for v in line.split()]
                rows.append(vals)
                if len(rows) == 3:
                    break
    if len(rows) != 3:
        return None
    M = np.eye(4)
    for i, row in enumerate(rows):
        M[i, :len(row)] = row
    return M


def _get_vox2ras(img):
    """Get vox→scanner-RAS affine from a nibabel image."""
    return np.array(img.affine)


def compute_seg_mask_mni_points(seg_mask_paths, subject_id, subjects_dir,
                                 threshold=0.5, subsample=None):
    """Transform seg mask nonzero voxels to MNI305 space.

    Chain: voxel (i,j,k) → scanner RAS (mask affine) → MNI305 (talairach.xfm)

    Args:
        seg_mask_paths:  list of NIfTI mask paths
        subject_id:      FreeSurfer subject ID
        subjects_dir:    FastSurfer subjects directory
        threshold:       mask value threshold
        subsample:       if set, subsample to this many points (for speed)

    Returns:
        mni_points:  (N, 3) ndarray of MNI305 coordinates, or None on failure
        n_voxels:    total nonzero voxels before subsampling
    """
    if not seg_mask_paths:
        return None, 0

    try:
        import nibabel as nib
    except ImportError:
        logging.warning("  nibabel not available for MNI seg mask evaluation")
        return None, 0

    # Read talairach.xfm (scanner RAS → MNI305)
    tal_xfm = _read_talairach_xfm(subjects_dir, subject_id)
    if tal_xfm is None:
        logging.warning(f"    No talairach.xfm for {subject_id}")
        return None, 0

    all_mni = []
    total_voxels = 0

    for mask_path in seg_mask_paths:
        if not pathlib.Path(mask_path).exists():
            logging.warning(f"    Seg mask not found: {mask_path}")
            continue

        try:
            mask_img = nib.load(mask_path)
            mask_data = np.squeeze(np.asarray(mask_img.dataobj))
        except Exception as e:
            logging.warning(f"    Failed to load seg mask {mask_path}: {e}")
            continue

        vox2ras = _get_vox2ras(mask_img)

        # Get all nonzero voxel coordinates
        nonzero = np.argwhere(mask_data > threshold)  # (N, 3) ijk
        total_voxels += len(nonzero)

        if len(nonzero) == 0:
            continue

        # Voxel → scanner RAS
        ones = np.ones((len(nonzero), 1))
        ijk_h = np.hstack([nonzero, ones])  # (N, 4)
        ras = (vox2ras @ ijk_h.T).T[:, :3]  # (N, 3)

        # Scanner RAS → MNI305
        ras_h = np.hstack([ras, ones])  # (N, 4)
        mni = (tal_xfm @ ras_h.T).T[:, :3]  # (N, 3)

        all_mni.append(mni)

        logging.info(f"    Seg mask → MNI: {len(nonzero)} voxels, "
                     f"mask shape={mask_data.shape}")

    if not all_mni:
        return None, total_voxels

    mni_points = np.concatenate(all_mni).astype(np.float32)

    # Optional subsampling for very large masks
    if subsample and len(mni_points) > subsample:
        idx = np.random.RandomState(42).choice(len(mni_points), subsample, replace=False)
        mni_points = mni_points[idx]

    return mni_points, total_voxels


def match_clusters_to_seg_mni(clusters, merged_pos, gt_mni_points,
                               match_distance=20.0):
    """Evaluate clusters against MNI-space ground truth points.

    MELD-style: a cluster is TP if ANY of its vertices (in fsaverage ≈ MNI space)
    is within match_distance of ANY ground truth voxel (in MNI space).

    Args:
        clusters:        list of cluster dicts (with 'vertices', 'centroid')
        merged_pos:      (N, 3) merged fsaverage positions (≈ MNI305)
        gt_mni_points:   (M, 3) MNI305 coordinates of seg mask voxels
        match_distance:  distance threshold in mm (MELD default: 20mm)

    Returns:
        cluster_eval dict with tp, fp, fn, sensitivity, ppv, etc.
    """
    gt_tree = cKDTree(gt_mni_points)
    gt_centroid = gt_mni_points.mean(axis=0)

    tp_clusters = set()
    for cl in clusters:
        cv = np.array(cl["vertices"])
        cluster_pos = merged_pos[cv]

        # Distance from each cluster vertex to nearest GT voxel
        dists, _ = gt_tree.query(cluster_pos)
        min_dist = float(dists.min())
        overlap = int((dists <= 1.0).sum())  # vertices within 1mm ≈ overlap

        cl["nearest_gt_mm"] = min_dist
        cl["gt_overlap"] = overlap

        if min_dist <= match_distance:
            tp_clusters.add(cl["cluster_id"])
            cl["is_tp"] = True
        else:
            cl["is_tp"] = False

    tp = len(tp_clusters)
    fp = len(clusters) - tp
    fn = 0 if tp > 0 else 1  # subject-level: was the lesion found at all?

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_annotations": 1,
        "sensitivity": float(tp > 0),
        "ppv": float(tp / max(len(clusters), 1)),
        "gt_mni_centroid": gt_centroid.tolist(),
        "gt_mni_n_voxels": len(gt_mni_points),
    }


def _save_mni_verification_nifti(gt_mni_points, clusters, merged_pos,
                                  out_dir, sid, seg_mask_paths=None,
                                  subjects_dir=None):
    """Save GT and cluster predictions as NIfTI volumes in MNI305 space.

    Creates volumes on a 1mm isotropic MNI grid:
      {sid}_gt_mni.nii.gz       — ground truth (seg mask voxels or annotation spheres)
      {sid}_clusters_mni.nii.gz — predicted clusters, each with unique label (rank)

    Also saves native-space files for verification:
      {sid}_gt_native.nii.gz    — original seg mask in native voxel space
      {sid}_t1_native.nii.gz    — subject T1 (orig.mgz) as NIfTI background

    All can be overlaid in any NIfTI viewer (fsleyes, freeview, ITK-SNAP).
    """
    try:
        import nibabel as nib
    except ImportError:
        return

    out_path = pathlib.Path(out_dir)

    # ── Copy native-space seg mask ──
    if seg_mask_paths:
        import shutil
        for mi, mp in enumerate(seg_mask_paths):
            if pathlib.Path(mp).exists():
                suffix = f"_{mi}" if len(seg_mask_paths) > 1 else ""
                dst = out_path / f"{sid}_gt_native{suffix}.nii.gz"
                if not dst.exists():
                    shutil.copy2(mp, str(dst))

    # Standard MNI305 grid: 1mm isotropic, covers [-100,100] x [-130,100] x [-80,110]
    x_range = (-100, 100)
    y_range = (-130, 100)
    z_range = (-80, 110)
    shape = (x_range[1] - x_range[0],
             y_range[1] - y_range[0],
             z_range[1] - z_range[0])

    # Affine: voxel (0,0,0) → MNI (-100, -130, -80)
    affine = np.eye(4)
    affine[0, 3] = x_range[0]
    affine[1, 3] = y_range[0]
    affine[2, 3] = z_range[0]

    # ── Save subject T1 resampled to MNI305 (same grid as GT + clusters) ──
    if subjects_dir:
        sd = pathlib.Path(subjects_dir) / sid
        t1_mni_dst = out_path / f"{sid}_t1_mni.nii.gz"
        if not t1_mni_dst.exists():
            tal_xfm = _read_talairach_xfm(subjects_dir, sid)
            for name in ["orig.mgz", "rawavg.mgz"]:
                orig_path = sd / "mri" / name
                if orig_path.exists() and tal_xfm is not None:
                    orig_img = nib.load(str(orig_path))
                    orig_data = np.asarray(orig_img.dataobj, dtype=np.float32)
                    orig_vox2ras = np.array(orig_img.affine)
                    # MNI → scanner RAS → native voxel
                    inv_tal = np.linalg.inv(tal_xfm)
                    inv_vox2ras = np.linalg.inv(orig_vox2ras)
                    mni2native_vox = inv_vox2ras @ inv_tal
                    # Build MNI grid coords and resample
                    from scipy.ndimage import map_coordinates
                    gi, gj, gk = np.mgrid[:shape[0], :shape[1], :shape[2]]
                    mni_coords = np.stack([
                        gi.ravel() + x_range[0],
                        gj.ravel() + y_range[0],
                        gk.ravel() + z_range[0],
                        np.ones(gi.size),
                    ])  # (4, N)
                    native_vox = mni2native_vox @ mni_coords  # (4, N)
                    t1_mni = map_coordinates(
                        orig_data, native_vox[:3], order=1, mode='constant', cval=0,
                    ).reshape(shape).astype(np.float32)
                    nib.save(nib.Nifti1Image(t1_mni, affine),
                             str(t1_mni_dst))
                    break


    def mni_to_vox(pts):
        """Convert MNI coords to voxel indices."""
        vox = pts.copy()
        vox[:, 0] -= x_range[0]
        vox[:, 1] -= y_range[0]
        vox[:, 2] -= z_range[0]
        return np.round(vox).astype(int)

    def in_bounds(ijk):
        return ((ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
                (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
                (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2]))

    # ── GT volume ──
    if gt_mni_points is not None and len(gt_mni_points) > 0:
        gt_vol = np.zeros(shape, dtype=np.float32)
        ijk = mni_to_vox(gt_mni_points)
        valid = in_bounds(ijk)
        vi = ijk[valid]
        gt_vol[vi[:, 0], vi[:, 1], vi[:, 2]] = 1.0
        nib.save(nib.Nifti1Image(gt_vol, affine),
                 str(out_path / f"{sid}_gt_mni.nii.gz"))

    # ── Cluster volume: each cluster gets its rank as label ──
    if clusters:
        cl_vol = np.zeros(shape, dtype=np.float32)
        for cl in clusters:
            cv = np.array(cl["vertices"])
            pts = merged_pos[cv]
            ijk = mni_to_vox(pts)
            valid = in_bounds(ijk)
            vi = ijk[valid]
            cl_vol[vi[:, 0], vi[:, 1], vi[:, 2]] = float(cl.get("rank", 1))
        nib.save(nib.Nifti1Image(cl_vol, affine),
                 str(out_path / f"{sid}_clusters_mni.nii.gz"))


# ── Hemisphere scoring ──

def _farthest_point_seeds(pos, n):
    N = pos.shape[0]; n = min(n, N)
    seeds = [0]; dists = np.full(N, np.inf)
    for _ in range(n - 1):
        dists = np.minimum(dists, np.linalg.norm(pos - pos[seeds[-1]], axis=1))
        seeds.append(int(np.argmax(dists)))
    return np.array(seeds)


@torch.no_grad()
def score_hemisphere(model, features, atlas_labels, mesh_cache,
                     norm_stats, device, patch_size=10000, num_patches=200, num_atlas_regions=36):
    """Score all vertices on one hemisphere. Returns dict with 'mean', 'variance', 'count'."""
    model.eval()
    ei = mesh_cache["edge_index"]; pos = mesh_cache["positions"]
    pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    adj = mesh_cache["adj_scipy"]

    if norm_stats.get("mean") is not None:
        features = (features - norm_stats["mean"]) / norm_stats["std"]

    cnt = np.zeros(N_FSAVERAGE_VERTICES, dtype=np.int32)
    mn = np.zeros(N_FSAVERAGE_VERTICES, dtype=np.float64)
    m2 = np.zeros(N_FSAVERAGE_VERTICES, dtype=np.float64)

    def upd(gi, sc):
        for g, s in zip(gi, sc):
            cnt[g] += 1; d = s - mn[g]; mn[g] += d / cnt[g]
            m2[g] += d * (s - mn[g])

    ns = min(max(num_patches, N_FSAVERAGE_VERTICES // patch_size + 10),
             N_FSAVERAGE_VERTICES)
    seeds = _farthest_point_seeds(pos_np, ns)
    from scipy.sparse.csgraph import breadth_first_order

    def _score_patch(seed):
        order, _ = breadth_first_order(adj, int(seed), directed=False)
        pg = order[:patch_size]; nl = len(pg)
        pn = torch.from_numpy(pg).long()
        ip = torch.zeros(N_FSAVERAGE_VERTICES, dtype=torch.bool); ip[pn] = True
        s, d = ei; keep = ip[s] & ip[d]
        mp = torch.full((N_FSAVERAGE_VERTICES,), -1, dtype=torch.long)
        mp[pn] = torch.arange(nl)
        lei = mp[ei[:, keep]]
        data = Data(
            x=features[pn], edge_index=lei,
            edge_attr=compute_edge_features(lei, pos[pn]),
            pos=pos[pn], atlas_label=atlas_labels[pn].clamp(0, num_atlas_regions - 1),
            num_nodes=nl,
        )
        batch = Batch.from_data_list([data]).to(device)
        return pn.numpy(), model.compute_anomaly_score(batch).cpu().numpy()

    for i in tqdm(range(len(seeds)), desc="    Tiling", leave=False):
        upd(*_score_patch(seeds[i]))

    # Fill gaps
    unc = (cnt == 0).sum()
    if unc > 0:
        ui = np.where(cnt == 0)[0]
        for seed in ui[np.random.choice(len(ui), min(max(20, unc//patch_size+5), len(ui)), replace=False)]:
            upd(*_score_patch(seed))

    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "mean": np.where(cnt > 0, mn, np.nan),
            "variance": np.where(cnt > 1, m2 / (cnt - 1), np.nan),
            "count": cnt,
        }

@torch.no_grad()
def score_hemisphere_per_feature(model, features, atlas_labels, mesh_cache,
                                 norm_stats, device, num_features_in=5,
                                 patch_size=5000, num_patches=200, num_atlas_regions=36):
    """Per-ORIGINAL-feature version of score_hemisphere. Returns (V, F): mean
    per-feature model error per vertex (NaN where unvisited). Col f = feature f."""
    model.eval()
    ei = mesh_cache["edge_index"]; pos = mesh_cache["positions"]
    pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    adj = mesh_cache["adj_scipy"]
    if norm_stats.get("mean") is not None:
        features = (features - norm_stats["mean"]) / norm_stats["std"]
    ssum = np.zeros((N_FSAVERAGE_VERTICES, num_features_in), dtype=np.float64)
    cnt  = np.zeros(N_FSAVERAGE_VERTICES, dtype=np.int32)
    ns = min(max(num_patches, N_FSAVERAGE_VERTICES // patch_size + 10), N_FSAVERAGE_VERTICES)
    seeds = _farthest_point_seeds(pos_np, ns)
    from scipy.sparse.csgraph import breadth_first_order

    def _patch(seed):
        order, _ = breadth_first_order(adj, int(seed), directed=False)
        pg = order[:patch_size]; nl = len(pg); pn = torch.from_numpy(pg).long()
        ip = torch.zeros(N_FSAVERAGE_VERTICES, dtype=torch.bool); ip[pn] = True
        s, d = ei; keep = ip[s] & ip[d]
        mp = torch.full((N_FSAVERAGE_VERTICES,), -1, dtype=torch.long); mp[pn] = torch.arange(nl)
        lei = mp[ei[:, keep]]
        data = Data(x=features[pn], edge_index=lei,
                    edge_attr=compute_edge_features(lei, pos[pn]), pos=pos[pn],
                    atlas_label=atlas_labels[pn].clamp(0, num_atlas_regions - 1), num_nodes=nl)
        batch = Batch.from_data_list([data]).to(device)
        return pg, model.compute_anomaly_score_per_feature(batch).cpu().numpy()

    for i in tqdm(range(len(seeds)), desc="    Tiling(feat)", leave=False):
        gi, pf = _patch(seeds[i]); ssum[gi] += pf; cnt[gi] += 1
    unc = int((cnt == 0).sum())                      # fill gaps so GT cores get a value
    if unc > 0:
        ui = np.where(cnt == 0)[0]
        for seed in ui[np.random.choice(len(ui), min(max(20, unc // patch_size + 5), len(ui)), replace=False)]:
            gi, pf = _patch(seed); ssum[gi] += pf; cnt[gi] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cnt[:, None] > 0, ssum / np.maximum(cnt[:, None], 1), np.nan)


# ── Z-score map from raw scores ──

def _compute_z_region_map(scores, atlas_labels, region_baselines, h_mean, h_std,
                          z_clip=100.0, vertex_median=None):
    """Per-vertex region-normalized z-score map. Skips region -1 (medial wall).
    Clips z-scores to [-z_clip, z_clip] to suppress outlier vertices.

    If `vertex_median` is provided (shape (N,), NaN where unavailable), it is
    used as the location estimate in the numerator instead of the per-region
    mean. The denominator stays region-level (MAD or std). Falls back to
    per-region mean for vertices where vertex_median is NaN. This captures
    within-region heterogeneity: a vertex whose healthy population is
    consistently slightly elevated above its region's mean gets a tighter
    null and only generates a z-score above threshold for patient-specific
    deviations, not for that systematic offset.
    """
    z = np.full(len(scores), np.nan, dtype=np.float32)
    vm = ~np.isnan(scores)
    use_vertex = vertex_median is not None
    for vidx in np.where(vm)[0]:
        rid_int = int(atlas_labels[vidx])
        if rid_int < 0:
            continue  # skip medial wall / unknown region
        rid = str(rid_int)
        rb = region_baselines.get(rid)
        # rb = None  # ← A6a GLOBAL ablation: single scalar h_mean/h_std everywhere; DELETE for default
        if rb is None or rb["std"] <= 1e-6:
            z[vidx] = (scores[vidx] - h_mean) / max(h_std, 1e-6)
            continue
        if use_vertex:
            vmed = vertex_median[vidx]
            loc = float(vmed) if not np.isnan(vmed) else rb["mean"]
        else:
            loc = rb["mean"]
        z[vidx] = (scores[vidx] - loc) / rb["std"]
    # Clip to suppress single-vertex outliers
    z[~np.isnan(z)] = np.clip(z[~np.isnan(z)], -z_clip, z_clip)
    return z


def _compute_mahalanobis_scores(raw_features, model_scores, atlas_labels,
                                 mahal_stats, clip=50.0):
    """Per-vertex Mahalanobis distance from healthy region distribution.

    For each vertex, constructs feature vector [thickness, curv, sulc, area,
    wg_pct, model_score] and computes Mahalanobis distance from the regional
    healthy mean using the inverse covariance matrix.

    Captures multivariate anomalies: e.g. thick cortex WITH low wg_pct
    (FCD signature) scores higher than thick cortex alone.

    Args:
        raw_features: (N, n_feat) raw morphometric features
        model_scores: (N,) raw model anomaly scores (before z-scoring)
        atlas_labels: (N,) atlas region labels
        mahal_stats: dict from baseline, per region: {mean, cov_inv, n}
        clip: max Mahalanobis distance to prevent outlier domination

    Returns:
        (N,) array of Mahalanobis distances per vertex
    """
    n_verts = len(raw_features)
    mahal_d = np.zeros(n_verts, dtype=np.float32)

    for rid_str, stats in mahal_stats.items():
        rid = int(rid_str)
        mu = np.array(stats["mean"])
        cov_inv = np.array(stats["cov_inv"])

        # Find vertices in this region with valid scores
        mask = (atlas_labels == rid) & ~np.isnan(model_scores)
        if mask.sum() == 0:
            continue

        # Build feature vectors: [raw_features, model_score]
        feat = raw_features[mask]  # (n, n_feat)
        sc = model_scores[mask].reshape(-1, 1)  # (n, 1)
        x = np.hstack([feat, sc])  # (n, n_feat+1)

        # Mahalanobis: sqrt((x-mu)^T @ cov_inv @ (x-mu)) for each vertex
        diff = x - mu  # (n, d)
        # Vectorized: (diff @ cov_inv) * diff, summed over features
        left = diff @ cov_inv  # (n, d)
        m2 = np.sum(left * diff, axis=1)  # (n,)
        m2 = np.clip(m2, 0, None)  # numerical safety
        mahal_d[mask] = np.sqrt(m2)

    # Clip extreme values
    mahal_d = np.clip(mahal_d, 0, clip)
    return mahal_d


def _combine_model_and_mahalanobis(model_z, mahal_d, weight=0.3):
    """Combine model z-score with Mahalanobis distance.

    Combined = model_z + weight * mahal_z

    where mahal_z is the Mahalanobis distance normalized to z-score scale
    (subtract median, divide by MAD to be robust to outliers).
    """
    combined = np.copy(model_z)
    valid = ~np.isnan(model_z) & (mahal_d > 0)

    if valid.sum() > 100:
        # Normalize Mahalanobis to z-score-like scale using robust stats
        md_valid = mahal_d[valid]
        median_d = np.median(md_valid)
        mad = np.median(np.abs(md_valid - median_d))
        mad = max(mad, 1e-6)
        mahal_z = (mahal_d - median_d) / (mad * 1.4826)  # 1.4826 = consistency factor for normal
        combined[valid] = model_z[valid] + weight * mahal_z[valid]

    return combined


# ── Visualization (whole-brain: both hemispheres combined) ──

CLUSTER_COLORS = ["#FF0000", "#00CC00", "#0066FF", "#FF9900", "#CC00FF",
                   "#00CCCC", "#FF3399", "#66FF00", "#3366CC", "#FF6600"]


def _save_combined_volume(hemi_data, sid, out_dir, subjects_dir):
    """Project both hemispheres into ONE anomaly_vol.nii.gz."""
    if not subjects_dir:
        return
    try:
        import nibabel as nib
        import nibabel.freesurfer as nfs
    except ImportError:
        return
    sd = pathlib.Path(subjects_dir) / sid
    for n in ["orig.mgz", "rawavg.mgz"]:
        op = sd / "mri" / n
        if op.exists():
            break
    else:
        return
    oi = nib.load(str(op)); vs = oi.shape[:3]; af = oi.affine
    r2v = np.linalg.inv(oi.header.get_vox2ras_tkr())

    vol = np.zeros(vs, dtype=np.float32)
    cv = np.zeros(vs, dtype=np.int32)

    for hemi, hd in hemi_data.items():
        sp = sd / "surf" / f"{hemi}.white"
        if not sp.exists():
            continue
        nc, _ = nfs.read_geometry(str(sp)); nn = nc.shape[0]
        scores = hd["scores"]
        if nn == len(scores):
            ns = np.nan_to_num(scores, nan=0).astype(np.float32)
        else:
            fsp = pathlib.Path(subjects_dir) / "fsaverage" / "surf" / f"{hemi}.sphere.reg"
            ssp = sd / "surf" / f"{hemi}.sphere.reg"
            if not fsp.exists() or not ssp.exists():
                continue
            fs, _ = nfs.read_geometry(str(fsp)); ss, _ = nfs.read_geometry(str(ssp))
            _, idx = cKDTree(fs).query(ss)
            ns = np.nan_to_num(scores[idx], nan=0).astype(np.float32)
        for vi in range(nn):
            s = ns[vi]
            if s == 0:
                continue
            vox = (r2v @ np.append(nc[vi], 1))[:3]
            i, j, k = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        ni, nj, nk = i+di, j+dj, k+dk
                        if 0 <= ni < vs[0] and 0 <= nj < vs[1] and 0 <= nk < vs[2]:
                            vol[ni, nj, nk] += s; cv[ni, nj, nk] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        vol = np.where(cv > 0, vol / cv, 0)
    nib.save(nib.Nifti1Image(vol, af), str(out_dir / f"{sid}_anomaly_vol.nii.gz"))


def _save_combined_anomaly_html(hemi_data, faces_cache, mapped_annotations,
                                 sid, out_dir):
    """Plotly HTML — both hemispheres in one interactive view."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    # Compute shared colorscale from both hemis
    all_valid = []
    for hd in hemi_data.values():
        v = hd["scores"][~np.isnan(hd["scores"])]
        if len(v) > 0:
            all_valid.append(v)
    if not all_valid:
        return
    all_valid = np.concatenate(all_valid)
    vmin, vmax = np.percentile(all_valid, 5), np.percentile(all_valid, 99)

    fig_data = []
    for hemi, hd in hemi_data.items():
        pos = hd["positions"]; scores = hd["scores"]
        vis = np.clip(np.nan_to_num(scores, nan=vmin), vmin, vmax)
        faces = faces_cache.get(hemi)
        if faces is not None:
            fig_data.append(go.Mesh3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=vis, colorscale="YlOrRd", cmin=vmin, cmax=vmax,
                colorbar=dict(title="Anomaly") if hemi == "lh" else None,
                opacity=0.9, name=f"{hemi} surface",
                showscale=(hemi == "lh"),
            ))

    if mapped_annotations:
        ax = [a["xyz_fsaverage"][0] for a in mapped_annotations]
        ay = [a["xyz_fsaverage"][1] for a in mapped_annotations]
        az = [a["xyz_fsaverage"][2] for a in mapped_annotations]
        fig_data.append(go.Scatter3d(
            x=ax, y=ay, z=az, mode="markers",
            marker=dict(size=20, color="blue", symbol="diamond",
                        line=dict(width=2, color="white")),
            name="Annotations"))

    fig = go.Figure(data=fig_data)
    fig.update_layout(title=f"Anomaly: {sid} (both hemispheres)",
                      scene=dict(aspectmode="data"), width=1400, height=900)
    fig.write_html(str(out_dir / f"{sid}_anomaly.html"))


def _save_combined_cluster_html(hemi_data, faces_cache, all_clusters,
                                 mapped_annotations, sid, out_dir,
                                 gt_vertices_merged=None):
    """Plotly HTML — clusters from both hemispheres in one view.
    Layering order (back to front):
      1. Brain surface (opaque anomaly map)
      2. GT lesion region (seg mask projection, green)
      3. Detected clusters (colored, foreground)
      4. Annotation points (if any)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    if not all_clusters:
        return

    # Build merged positions for cluster vertex lookup
    merged_pos_parts = []
    for hemi in sorted(hemi_data.keys()):
        merged_pos_parts.append(hemi_data[hemi]["positions"])
    merged_pos = np.concatenate(merged_pos_parts)

    # Shared z colorscale
    all_z = []
    for hd in hemi_data.values():
        z = np.nan_to_num(hd["z_map"], nan=0)
        v = z[z != 0]
        if len(v) > 0:
            all_z.append(v)
    if not all_z:
        return
    all_z = np.concatenate(all_z)
    vmin, vmax = np.percentile(all_z, 5), np.percentile(all_z, 99)

    fig_data = []

    # ── Layer 1: Brain surface (opaque anomaly map, background) ──
    for hemi, hd in hemi_data.items():
        pos = hd["positions"]; z_clean = np.nan_to_num(hd["z_map"], nan=0)
        faces = faces_cache.get(hemi)
        if faces is not None:
            fig_data.append(go.Mesh3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=np.clip(z_clean, vmin, vmax), colorscale="RdYlGn_r",
                cmin=vmin, cmax=vmax, opacity=1.0, name=f"{hemi} z-region",
                showscale=(hemi == "lh"),
            ))

    # ── Layer 2: GT lesion region (seg mask, green, elevated) ──
    if gt_vertices_merged is not None and len(gt_vertices_merged) > 0:
        gt_v = np.array(list(gt_vertices_merged))
        fig_data.append(go.Scatter3d(
            x=merged_pos[gt_v, 0], y=merged_pos[gt_v, 1], z=merged_pos[gt_v, 2],
            mode="markers",
            marker=dict(size=3, color="#00DD00", opacity=1.0,
                        line=dict(width=0)),
            name=f"GT lesion ({len(gt_v)} vertices)"))

    # ── Layer 2b: Point annotations (blue diamonds) ──
    if mapped_annotations:
        ax = [a["xyz_fsaverage"][0] for a in mapped_annotations
              if "lesion_vertices" not in a]
        ay = [a["xyz_fsaverage"][1] for a in mapped_annotations
              if "lesion_vertices" not in a]
        az = [a["xyz_fsaverage"][2] for a in mapped_annotations
              if "lesion_vertices" not in a]
        if ax:
            fig_data.append(go.Scatter3d(
                x=ax, y=ay, z=az, mode="markers",
                marker=dict(size=20, color="blue", symbol="diamond",
                            line=dict(width=2, color="white")),
                name="Annotations"))

    # ── Layer 3: Detected clusters (foreground, bright colors) ──
    for ci, cl in enumerate(all_clusters):
        cv = np.array(cl["vertices"])
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        hemi_tag = cl.get("hemi", "?")
        tp_tag = " TP" if cl.get("is_tp") else ""
        fig_data.append(go.Scatter3d(
            x=merged_pos[cv, 0], y=merged_pos[cv, 1], z=merged_pos[cv, 2],
            mode="markers",
            marker=dict(size=4, color=color, opacity=1.0,
                        line=dict(width=0.5, color="white")),
            name=f"C{cl['rank']} {hemi_tag}{tp_tag} "
                 f"({cl['area_cm2']:.1f}cm², z={cl['peak_z']:.1f})"))
        centroid = np.array(cl["centroid"])
        fig_data.append(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode="markers+text",
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(width=1, color="white")),
            text=[f"C{cl['rank']}"], textfont=dict(size=12, color="white"),
            showlegend=False))

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title=f"Clusters: {sid} ({len(all_clusters)} found)",
        scene=dict(aspectmode="data"),
        width=1400, height=900,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_dir / f"{sid}_clusters.html"))


# ═══════════════════════════════════════════════════════════════
#  Cluster detection + matching
# ═══════════════════════════════════════════════════════════════

FSAVERAGE_VERTEX_AREA_MM2 = 1.2


def find_clusters(z_map, edge_index, positions, threshold=1.5,
                  min_vertices=10, max_vertices=3000, max_clusters=10,
                  atlas_labels=None, sort_by="area_mean_z",
                  exclude_regions=None):
    from collections import deque
    N = len(z_map)
    above = np.array([not np.isnan(z_map[i]) and z_map[i] > threshold for i in range(N)])
    if above.sum() == 0:
        return []

    adj = {}
    s, d = edge_index[0].numpy(), edge_index[1].numpy()
    for a, b in zip(s, d):
        if above[a] and above[b]:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

    visited = np.zeros(N, dtype=bool); raw = []
    for seed in np.where(above)[0]:
        if visited[seed]:
            continue
        comp = []; q = deque([seed]); visited[seed] = True
        while q:
            v = q.popleft(); comp.append(v)
            for nb in adj.get(v, []):
                if not visited[nb]:
                    visited[nb] = True; q.append(nb)
        raw.append(np.array(comp))

    clusters = []
    for cid, verts in enumerate(raw):
        if len(verts) < min_vertices or len(verts) > max_vertices:
            continue
        zv = z_map[verts]
        c = {
            "cluster_id": cid, "vertices": verts.tolist(),
            "n_vertices": len(verts),
            "area_cm2": float(len(verts) * FSAVERAGE_VERTEX_AREA_MM2 / 100),
            "centroid": positions[verts].mean(0).tolist(),
            "mean_z": float(zv.mean()), "peak_z": float(zv.max()),
            "peak_vertex": int(verts[np.argmax(zv)]),
            "confidence": float((zv > 2).sum() / len(zv)),
        }
        if atlas_labels is not None:
            rids, cnts = np.unique(atlas_labels[verts], return_counts=True)
            si = np.argsort(-cnts)
            c["atlas_regions"] = [{"region_id": int(rids[i]), "n": int(cnts[i])} for i in si[:3]]
        clusters.append(c)

    if sort_by == "mean_z":
        clusters.sort(key=lambda c: c["mean_z"], reverse=True)
    elif sort_by == "area_mean_z":
        clusters.sort(key=lambda c: c["area_cm2"] * c["mean_z"], reverse=True)
    else:  # peak_z
        clusters.sort(key=lambda c: c["peak_z"], reverse=True)

    # Filter artifact regions BEFORE max_clusters cutoff
    if exclude_regions and atlas_labels is not None:
        exclude_set = set(exclude_regions)
        n_before = len(clusters)
        clusters = [c for c in clusters
                    if not (c.get("atlas_regions")
                            and c["atlas_regions"][0]["region_id"] in exclude_set)]
        n_filtered = n_before - len(clusters)
        if n_filtered > 0:
            logging.info(f"    Filtered {n_filtered} artifact clusters "
                         f"(regions {sorted(exclude_set)})")

    clusters = clusters[:max_clusters]
    for r, c in enumerate(clusters):
        c["rank"] = r + 1
    return clusters


# ═══════════════════════════════════════════════════════════════
#  BASELINE: Score healthy subjects, save stats
# ═══════════════════════════════════════════════════════════════

def run_healthy_baseline(args, dm):
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.val_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dataset = getattr(args, 'baseline_dataset', 'eval')
    baseline_split = getattr(args, 'baseline_split', 'val')

    if baseline_dataset == 'train':
        norm_stats = dm.train_dataset.get_norm_stats()
        if baseline_split == 'val' and hasattr(dm, 'train_val_dicts') and dm.train_val_dicts:
            baseline_ds = dm._build_dataset(dm.train_val_dicts,
                                            norm_stats=norm_stats,
                                            subjects_dir_override=dm.args.subjects_dir_ext)
            healthy = [s for s in baseline_ds.samples if not s["has_anomaly"]]
        elif baseline_split == 'test' and hasattr(dm, 'train_test_dicts') and dm.train_test_dicts:
            baseline_ds = dm._build_dataset(dm.train_test_dicts,
                                            norm_stats=norm_stats,
                                            subjects_dir_override=dm.args.subjects_dir_ext)
            healthy = [s for s in baseline_ds.samples if not s["has_anomaly"]]
        elif baseline_split == 'train':
            healthy = [s for s in dm.train_dataset.samples if not s["has_anomaly"]]
        else:
            logging.warning(f"  baseline_split={baseline_split} not available for train dataset, "
                            f"falling back to train")
            healthy = [s for s in dm.train_dataset.samples if not s["has_anomaly"]]
        logging.info(f"  Baseline: train dataset, {baseline_split} split ({len(healthy)} samples)")
    else:  # baseline_dataset == 'eval'
        norm_stats = dm.val_dataset.get_norm_stats()
        if baseline_split == 'val':
            healthy = [s for s in dm.val_dataset.samples if not s["has_anomaly"]]
        elif baseline_split == 'test':
            healthy = [s for s in dm.test_dataset.samples if not s["has_anomaly"]]
        elif baseline_split == 'train':
            if hasattr(dm, 'eval_train_dicts') and dm.eval_train_dicts:
                baseline_ds = dm._build_dataset(dm.eval_train_dicts,
                                                norm_stats=norm_stats)
                healthy = [s for s in baseline_ds.samples if not s["has_anomaly"]]
            else:
                healthy = [s for s in dm.train_dataset.samples if not s["has_anomaly"]]
        else:
            healthy = [s for s in dm.val_dataset.samples if not s["has_anomaly"]]
        logging.info(f"  Baseline: eval dataset, {baseline_split} split ({len(healthy)} samples)")

    model = _load_model(args, device)
    _prepare_meshes(dm)
    n = min(len(healthy), args.max_healthy)

    logging.info(f"\n=== Healthy baseline ({n} hemisphere-samples) ===")
    n_features = healthy[0]["features"].shape[1] if n > 0 else 0

    # ── Pass 1: score every hemisphere-sample, store full per-vertex arrays.
    #    We need per-sample medians first to detect subject-level outliers
    #    before aggregating into region statistics.
    per_sample_scores = []   # full (V,) score arrays per sample, NaN where invalid
    per_sample_atlas = []
    per_sample_feat = []
    per_sample_id = []       # (subject_id, hemi) tuples
    per_sample_median = []   # scalar median per sample (for outlier filter)

    for i, s in enumerate(healthy[:n]):
        sid, hemi = s["subject_id"], s["hemi"]
        logging.info(f"  [{i+1}/{n}] {sid} {hemi}")
        r = score_hemisphere(
            model, s["features"], s["atlas_labels"],
            dm.mesh_cache[hemi], norm_stats, device,
            patch_size=args.patch_size, num_patches=args.num_patches_healthy,
            num_atlas_regions=args.num_atlas_regions,
        )
        sc = r["mean"]
        vm = ~np.isnan(sc)
        v = sc[vm]
        per_sample_scores.append(sc)
        per_sample_atlas.append(s["atlas_labels"].numpy())
        per_sample_feat.append(s["features"].numpy())
        per_sample_id.append((sid, hemi))
        if v.size > 0:
            med_i = float(np.median(v))
            mad_i = float(median_abs_deviation(v, scale='normal'))
        else:
            med_i, mad_i = float('nan'), float('nan')
        per_sample_median.append(med_i)
        logging.info(f"    median={med_i:.4f}  MAD={mad_i:.4f}  ({v.size} verts)")

    # ── Subject outlier filter: identify healthy samples whose median is
    #    far from the population median (likely scan-quality / preprocessing
    #    artifacts). Threshold = 3 × MAD on the distribution of per-sample
    #    medians. Excluded samples don't contribute to region or vertex stats.
    OUTLIER_THRESH = 3.0
    sample_medians = np.array(per_sample_median, dtype=np.float64)
    valid = ~np.isnan(sample_medians)
    if valid.sum() >= 4:
        pop_med = float(np.median(sample_medians[valid]))
        pop_mad = float(median_abs_deviation(sample_medians[valid], scale='normal'))
    else:
        pop_med, pop_mad = float('nan'), 0.0

    if pop_mad > 1e-6 and valid.sum() >= 4:
        deviation = np.abs(sample_medians - pop_med)
        keep_mask = valid & (deviation <= OUTLIER_THRESH * pop_mad)
        excluded = [(per_sample_id[i], sample_medians[i])
                    for i in range(n) if not keep_mask[i]]
        if excluded:
            logging.info(
                f"\n  Outlier filter: excluding {len(excluded)}/{n} hemisphere-samples "
                f"(|median - {pop_med:.4f}| > {OUTLIER_THRESH} × {pop_mad:.4f})"
            )
            for (sid, hemi), m in excluded:
                z = abs(m - pop_med) / pop_mad if not np.isnan(m) else float('inf')
                logging.info(f"    EXCLUDE {sid} {hemi}: median={m:.4f}  (|z|={z:.2f})")
        else:
            logging.info(
                f"\n  Outlier filter: no exclusions "
                f"(population median={pop_med:.4f}, MAD={pop_mad:.4f})"
            )
    else:
        logging.info(
            f"\n  Outlier filter: skipped "
            f"(n_valid={int(valid.sum())}, MAD={pop_mad:.6f})"
        )
        keep_mask = valid
        excluded = []

    n_kept = int(keep_mask.sum())
    excluded_records = [
        {"subject_id": sid, "hemi": hemi,
         "median": float(m) if not np.isnan(m) else None}
        for (sid, hemi), m in excluded
    ]

    # ── Pass 2: aggregate kept samples into global + per-region + Mahalanobis.
    #    Field names "mean"/"std" preserved for downstream compatibility,
    #    but values are now MEDIAN (location) and MAD×1.4826 (scale).
    all_scores = []
    region_scores = defaultdict(list)
    region_feature_vecs = {}  # region → list of (n_verts, n_features+1) blocks

    for i in range(n):
        if not keep_mask[i]:
            continue
        sc = per_sample_scores[i]
        atlas = per_sample_atlas[i]
        raw_feat = per_sample_feat[i]
        vm = ~np.isnan(sc)
        all_scores.extend(sc[vm].tolist())
        for rid in np.unique(atlas):
            rid = int(rid)
            rm = (atlas == rid) & vm
            if rm.sum() > 0:
                region_scores[rid].extend(sc[rm].tolist())
                feat_block = raw_feat[rm]
                score_block = sc[rm].reshape(-1, 1)
                combined = np.hstack([feat_block, score_block])
                region_feature_vecs.setdefault(rid, []).append(combined)

    def robust_stats(x):
        """Median + MAD×1.4826 (Gaussian-consistent), with scale floor."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0, 1.0
        med = float(np.median(x))
        mad = float(median_abs_deviation(x, scale='normal'))
        return med, max(mad, 1e-3)

    g_med, g_mad = robust_stats(all_scores)
    rb = {}
    for k, v in sorted(region_scores.items()):
        med, mad = robust_stats(v)
        rb[str(k)] = {"mean": med, "std": mad, "n": len(v)}

    # ── Mahalanobis (multivariate): standard mean/cov on filtered samples.
    #    Subject-level outliers were already removed, so estimates aren't
    #    driven by pathological hemisphere-samples.
    mahal_stats = {}
    for rid, blocks in sorted(region_feature_vecs.items()):
        data = np.vstack(blocks)
        if len(data) < 50:
            continue
        mu = data.mean(axis=0)
        cov = np.cov(data, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
            mahal_stats[str(rid)] = {
                "mean": mu.tolist(),
                "cov_inv": cov_inv.tolist(),
                "n": len(data),
            }
        except np.linalg.LinAlgError:
            logging.warning(f"    Region {rid}: singular covariance, skipping Mahalanobis")

    logging.info(f"  Mahalanobis stats: {len(mahal_stats)} regions")

    baseline = {
        "global_mean": g_med,
        "global_std": g_mad,
        "n_vertices": len(all_scores),
        "n_subjects": n_kept,
        "n_subjects_total": n,
        "n_subjects_excluded": len(excluded_records),
        "excluded_subjects": excluded_records,
        "outlier_threshold_mad": OUTLIER_THRESH,
        "estimator": "median_mad",
        "region_baselines": rb,
        "mahalanobis": mahal_stats,
        "n_features": n_features,
    }
    bp = out_dir / "healthy_baseline.json"
    with open(bp, "w") as f:
        json.dump(baseline, f, indent=2)
    logging.info(f"\n  Saved: {bp}")
    logging.info(
        f"  Global: median={baseline['global_mean']:.4f}  "
        f"MAD={baseline['global_std']:.4f}  "
        f"({n_kept}/{n} hemisphere-samples kept, {len(rb)} regions)"
    )

    # ── Per-vertex baseline: median across kept healthy samples, per hemi.
    #    Saved as a separate .npz (bulky, JSON-unfriendly). The detection
    #    z-score code uses this as the location estimate (numerator) while
    #    keeping per-region MAD as the scale (denominator) — captures
    #    within-region heterogeneity without the noise of per-vertex scale.
    hemi_score_lists = {}
    hemi_sample_counts = {}
    for i in range(n):
        if not keep_mask[i]:
            continue
        _, hemi = per_sample_id[i]
        hemi_score_lists.setdefault(hemi, []).append(per_sample_scores[i])
        hemi_sample_counts[hemi] = hemi_sample_counts.get(hemi, 0) + 1

    vertex_arrays = {}
    for hemi, sc_list in hemi_score_lists.items():
        mat = np.stack(sc_list, axis=0)  # (n_kept_hemi_samples, n_vertices)
        with np.errstate(all='ignore'):
            med = np.nanmedian(mat, axis=0).astype(np.float32)
        coverage = (~np.isnan(mat)).sum(axis=0)
        # Require ≥4 healthy samples to trust a per-vertex median; mark
        # others NaN so z-score code falls back to per-region for them.
        med[coverage < 4] = np.nan
        vertex_arrays[f"{hemi}_median"] = med
        vertex_arrays[f"{hemi}_coverage"] = coverage.astype(np.int16)
        n_trusted = int((~np.isnan(med)).sum())
        logging.info(
            f"  Per-vertex {hemi}: {hemi_sample_counts[hemi]} samples, "
            f"{n_trusted}/{len(med)} vertices with ≥4 coverage"
        )

    if vertex_arrays:
        vp = out_dir / "healthy_baseline_vertex.npz"
        np.savez_compressed(str(vp), **vertex_arrays)
        logging.info(f"  Saved per-vertex baseline: {vp}")


# ═══════════════════════════════════════════════════════════════
#  FP ANALYSIS: Find systematic artifact regions on healthy brains
# ═══════════════════════════════════════════════════════════════

def run_fp_analysis(args, dm):
    """
    Score ALL healthy subjects, run cluster detection on each,
    count which atlas regions consistently produce clusters.
    Regions that appear in >50% of healthy subjects are artifact hotspots.

    Saves: fp_regions.json with region IDs to exclude from detection.
    """
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.val_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline(args)
    h_mean, h_std = baseline["global_mean"], baseline["global_std"]
    rb = baseline["region_baselines"]

    norm_stats = dm.val_dataset.get_norm_stats()
    model = _load_model(args, device)
    _prepare_meshes(dm)

    eval_ds = dm.val_dataset if args.split == "val" else dm.test_dataset

    # Group healthy subjects
    subj_samples = defaultdict(list)
    for s in eval_ds.samples:
        if not s["has_anomaly"]:
            subj_samples[s["subject_id"]].append(s)

    n_subjects = len(subj_samples)
    logging.info(f"\n{'='*60}")
    logging.info(f"FP ANALYSIS — {n_subjects} healthy subjects")
    logging.info(f"{'='*60}")

    # Count: how many subjects have a cluster in each region
    region_subject_count = defaultdict(int)  # region_id → n_subjects with cluster there
    region_cluster_count = defaultdict(int)  # region_id → total clusters across all subjects
    region_cluster_details = defaultdict(list)  # region_id → list of (peak_z, area_cm2)

    for si, (sid, samples) in enumerate(subj_samples.items()):
        logging.info(f"  [{si+1}/{n_subjects}] {sid}")

        # Score both hemis and merge
        hemi_data = {}
        for sample in samples:
            hemi = sample["hemi"]
            mesh = dm.mesh_cache[hemi]
            r = score_hemisphere(
                model, sample["features"], sample["atlas_labels"],
                dm.mesh_cache[hemi], norm_stats, device,
                patch_size=args.patch_size, num_patches=args.num_patches_healthy,
                num_atlas_regions=args.num_atlas_regions,
            )
            atlas_np = sample["atlas_labels"].numpy()
            z_map = _compute_z_region_map(r["mean"], atlas_np, rb, h_mean, h_std)
            hemi_data[hemi] = {
                "scores": r["mean"], "z_map": z_map,
                "positions": mesh["positions"].numpy(),
                "atlas_labels": atlas_np,
                "edge_index": mesh["edge_index"],
            }

        # Merge hemis
        merged_z, merged_pos, merged_edges_list, merged_atlas = [], [], [], []
        offset = 0
        for hemi in sorted(hemi_data.keys()):
            hd = hemi_data[hemi]
            n = len(hd["z_map"])
            merged_z.append(hd["z_map"])
            merged_pos.append(hd["positions"])
            merged_atlas.append(hd["atlas_labels"])
            ei = hd["edge_index"].clone(); ei += offset
            merged_edges_list.append(ei)
            offset += n

        merged_z = np.concatenate(merged_z)
        merged_pos = np.concatenate(merged_pos)
        merged_atlas = np.concatenate(merged_atlas)
        merged_edges = torch.cat(merged_edges_list, dim=1)

        # Find clusters (same params as detection)
        clusters = find_clusters(
            merged_z, merged_edges, merged_pos,
            threshold=args.cluster_z_threshold,
            min_vertices=args.min_cluster_vertices,
            max_vertices=getattr(args, "max_cluster_vertices", 3000),
            max_clusters=50,  # find more to get full picture
            atlas_labels=merged_atlas,
        )

        # Track which regions produced clusters for this subject
        regions_this_subject = set()
        for cl in clusters:
            if "atlas_regions" in cl and cl["atlas_regions"]:
                primary_rid = cl["atlas_regions"][0]["region_id"]
                regions_this_subject.add(primary_rid)
                region_cluster_count[primary_rid] += 1
                region_cluster_details[primary_rid].append({
                    "peak_z": cl["peak_z"], "area_cm2": cl["area_cm2"],
                    "n_vertices": cl["n_vertices"],
                })

        for rid in regions_this_subject:
            region_subject_count[rid] += 1

        logging.info(f"    {len(clusters)} clusters, regions: "
                     f"{sorted(regions_this_subject)}")

    # ── Compute per-region FP rate ──
    fp_threshold = 0.7  # region appears in >30% of healthy subjects → artifact
    fp_regions = []
    all_region_stats = []

    logging.info(f"\n{'='*60}")
    logging.info(f"REGION FP RATES (threshold: {fp_threshold:.0%} of healthy subjects)")
    logging.info(f"{'='*60}")

    for rid in sorted(region_subject_count.keys()):
        n_subj = region_subject_count[rid]
        rate = n_subj / n_subjects
        n_cl = region_cluster_count[rid]
        details = region_cluster_details[rid]
        avg_peak_z = np.mean([d["peak_z"] for d in details]) if details else 0

        stat = {
            "region_id": rid,
            "n_subjects_with_cluster": n_subj,
            "fp_rate": float(rate),
            "total_clusters": n_cl,
            "avg_peak_z": float(avg_peak_z),
            "is_artifact": rate >= fp_threshold,
        }
        all_region_stats.append(stat)

        flag = " ← ARTIFACT" if rate >= fp_threshold else ""
        logging.info(f"  Region {rid:>3d}: {rate:5.1%} ({n_subj}/{n_subjects} subjects), "
                     f"{n_cl} clusters, avg_peak_z={avg_peak_z:.1f}{flag}")

        if rate >= fp_threshold:
            fp_regions.append(rid)

    # Save
    result = {
        "n_healthy_subjects": n_subjects,
        "fp_threshold": fp_threshold,
        "fp_regions": fp_regions,
        "region_stats": all_region_stats,
    }
    fp_path = out_dir / "fp_regions.json"
    with open(fp_path, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"\n  Artifact regions to exclude: {fp_regions}")
    logging.info(f"  Saved: {fp_path}")


def _load_fp_regions(args) -> List[int]:
    """Load FP region exclusion list. Returns empty list if not available."""
    fp_path = pathlib.Path(args.val_output_dir) / "fp_regions.json"
    if not fp_path.exists():
        return []
    with open(fp_path) as f:
        data = json.load(f)
    return data.get("fp_regions", [])


# ═══════════════════════════════════════════════════════════════
#  DETECTION: Lesional MRIs only → cluster-level sens + PPV
# ═══════════════════════════════════════════════════════════════

def run_detection(args, dm):
    """
    Score ONLY lesional subjects. For each:
      - Score both hemispheres
      - Find clusters on z-region map
      - Match clusters to the segmentation-mask ground truth (MNI305)
    Reports aggregate MELD-style sensitivity + PPV.
    """
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.val_output_dir) / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline(args)
    h_mean, h_std = baseline["global_mean"], baseline["global_std"]
    rb = baseline["region_baselines"]
    vertex_baselines = baseline.get("vertex_baselines", {}) or {}
    mahal = baseline.get("mahalanobis", {})
    use_ensemble = getattr(args, "ensemble_raw", False) and mahal
    raw_weight = getattr(args, "raw_weight", 0.3)
    if vertex_baselines:
        logging.info(
            f"  Using per-vertex location: "
            f"{', '.join(sorted(vertex_baselines.keys()))} "
            f"(per-region scale)"
        )
    if use_ensemble:
        logging.info(f"  Ensemble: model + Mahalanobis (weight={raw_weight}, {len(mahal)} regions)")
    elif getattr(args, "ensemble_raw", False) and not mahal:
        logging.warning(f"  --ensemble_raw requested but no mahalanobis in baseline. "
                        f"Re-run baseline to generate Mahalanobis stats.")

    # Load FP region exclusion list (from file + CLI --exclude_regions)
    fp_regions = set(_load_fp_regions(args)) | set(getattr(args, 'exclude_regions', []))

    norm_stats = dm.val_dataset.get_norm_stats()
    model = _load_model(args, device)
    _prepare_meshes(dm)

    # Use dm.subjects_dir (the external FastSurfer subjects dir).
    subjects_dir = dm.subjects_dir
    faces_cache = _load_faces(dm, subjects_dir)

    eval_ds = dm.val_dataset if args.split == "val" else dm.test_dataset

    # Group by subject, keep only lesional
    subj_samples = defaultdict(list)
    for s in eval_ds.samples:
        subj_samples[s["subject_id"]].append(s)
    patho_subjects = {sid: slist for sid, slist in subj_samples.items()
                      if any(s["has_anomaly"] for s in slist)}

    if fp_regions:
        logging.info(f"  Excluding artifact regions: {sorted(fp_regions)}")

    logging.info(f"\n{'='*60}")
    logging.info(f"DETECTION — {len(patho_subjects)} lesional subjects")
    logging.info(f"{'='*60}")

    all_results = []

    for si, (sid, samples) in enumerate(patho_subjects.items()):
        # Ground truth is a volumetric segmentation mask per subject.
        seg_masks = samples[0].get("seg_masks", [])
        has_seg = len(seg_masks) > 0

        if not has_seg:
            logging.warning(f"    {sid}: has_anomaly=True but no seg_masks! "
                            f"Keys: {list(samples[0].keys())}")

        gt_desc = f"{len(seg_masks)} seg masks" if has_seg else "no ground truth"

        logging.info(f"\n  [{si+1}/{len(patho_subjects)}] {sid} ({gt_desc})")

        sdir = out_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)

        hemi_data = {}

        # ── Score both hemispheres ──
        for sample in samples:
            hemi = sample["hemi"]
            mesh = dm.mesh_cache[hemi]
            pos_np = mesh["positions"].numpy()

            r = score_hemisphere(
                model, sample["features"], sample["atlas_labels"],
                dm.mesh_cache[hemi], norm_stats, device,
                patch_size=args.patch_size, num_patches=args.num_patches,
                num_atlas_regions=args.num_atlas_regions,
            )
            atlas_np = sample["atlas_labels"].numpy()
            vbh = vertex_baselines.get(hemi)
            vertex_median = vbh["median"] if vbh else None
            z_map = _compute_z_region_map(r["mean"], atlas_np, rb, h_mean, h_std,
                                          vertex_median=vertex_median)

            # Ensemble: combine model z with Mahalanobis distance
            if use_ensemble:
                raw_feat = sample["features"].numpy()
                mahal_d = _compute_mahalanobis_scores(
                    raw_feat, r["mean"], atlas_np, mahal)
                z_map = _combine_model_and_mahalanobis(z_map, mahal_d, raw_weight)

            hemi_data[hemi] = {
                "scores": r["mean"], "z_map": z_map, "positions": pos_np,
                "atlas_labels": atlas_np,
                "edge_index": mesh["edge_index"],
            }

        # Save z-maps for ensemble use
        for hemi, hd in hemi_data.items():
            np.save(str(sdir / f"{hemi}_zmap.npy"), hd["z_map"])

        # ── Map ground truth ──
        # The public datasets (FCD Bonn, IDEAS) ship a volumetric segmentation
        # mask per lesional subject. We transform its nonzero voxels into MNI305
        # space and match clusters there (no surface projection needed).
        all_mapped = []          # kept empty; only used by the surface viewers
        gt_mni_points = None     # MNI-space GT points for seg-mask evaluation

        if has_seg:
            gt_mni_points, gt_n_voxels = compute_seg_mask_mni_points(
                seg_masks, sid, subjects_dir,
            )
            if gt_mni_points is not None:
                logging.info(f"    Seg mask → {len(gt_mni_points)} MNI points "
                             f"(from {gt_n_voxels} voxels)")
            else:
                logging.warning(f"    Failed to compute MNI points for {sid}")

        # ── Merge both hemis into whole-brain arrays ──
        merged_z = []
        merged_pos = []
        merged_edges_list = []
        merged_atlas = []
        hemi_offset = {}
        offset = 0

        for hemi in sorted(hemi_data.keys()):
            hd = hemi_data[hemi]
            n = len(hd["z_map"])
            hemi_offset[hemi] = offset
            merged_z.append(hd["z_map"])
            merged_pos.append(hd["positions"])
            merged_atlas.append(hd["atlas_labels"])
            # Shift edge indices by offset
            ei = hd["edge_index"].clone()
            ei += offset
            merged_edges_list.append(ei)
            offset += n

        merged_z = np.concatenate(merged_z)
        merged_pos = np.concatenate(merged_pos)
        merged_atlas = np.concatenate(merged_atlas)
        merged_edges = torch.cat(merged_edges_list, dim=1)

        # ── Find clusters on whole brain ──
        clusters = find_clusters(
            merged_z, merged_edges, merged_pos,
            threshold=args.cluster_z_threshold,
            min_vertices=args.min_cluster_vertices,
            max_vertices=getattr(args, "max_cluster_vertices", 3000),
            max_clusters=args.max_clusters,
            atlas_labels=merged_atlas,
            sort_by=getattr(args, "cluster_sort", "area_mean_z"),
            exclude_regions=fp_regions,
        )

        # Tag each cluster with hemisphere
        for c in clusters:
            pv = c["peak_vertex"]
            for hemi, ho in hemi_offset.items():
                n_hemi = len(hemi_data[hemi]["z_map"])
                if ho <= pv < ho + n_hemi:
                    c["hemi"] = hemi
                    break
            else:
                c["hemi"] = "lh" if c["centroid"][0] < 0 else "rh"

        # ── Match clusters to ground truth (seg mask in MNI305) ──
        if has_seg and gt_mni_points is not None:
            # MNI-space matching: compare cluster vertices (in fsaverage ≈ MNI)
            # directly against seg mask voxels transformed to MNI305.
            # No surface projection needed — works for subcortical lesions too.
            cluster_eval = match_clusters_to_seg_mni(
                clusters, merged_pos, gt_mni_points,
                match_distance=args.cluster_match_distance,
            )
        else:
            # Seg mask present but MNI transform failed — cannot evaluate
            logging.warning(f"    Cannot evaluate {sid}: seg mask but no talairach.xfm")
            cluster_eval = {
                "tp": 0, "fp": len(clusters), "fn": 1,
                "n_annotations": 1, "sensitivity": 0.0,
                "ppv": 0.0, "eval_failed": True,
            }

        # ── Log results (unified for all datasets) ──
        if clusters:
            logging.info(f"    {len(clusters)} clusters, "
                         f"TP={cluster_eval['tp']} FP={cluster_eval['fp']} "
                         f"FN={cluster_eval['fn']}")
            for c in clusters:
                tag = "TP" if c.get("is_tp") else "FP"
                best_d = c.get("nearest_gt_mm", float('inf'))
                overlap_str = f" ov={c.get('gt_overlap', 0)}" if has_seg else ""
                region_str = f" r{c['atlas_regions'][0]['region_id']}" if c.get('atlas_regions') else ""
                logging.info(f"      C{c['rank']:>2d} ({c.get('hemi', '?')}) "
                             f"[{tag}] {c['n_vertices']:>3d}v "
                             f"z={c['peak_z']:>5.1f} "
                             f"{c['area_cm2']:.1f}cm² "
                             f"d={best_d:.0f}mm{region_str}{overlap_str}")
        else:
            logging.info(f"    No clusters found")

        # ── Save outputs (all whole-brain combined) ──
        _save_combined_volume(hemi_data, sid, sdir, subjects_dir)
        _save_combined_anomaly_html(hemi_data, faces_cache, [], sid, sdir)
        _save_combined_cluster_html(hemi_data, faces_cache, clusters,
                                     [], sid, sdir, gt_vertices_merged=None)

        # ── MNI verification NIfTI (GT + clusters in same space) ──
        _save_mni_verification_nifti(gt_mni_points, clusters, merged_pos, sdir, sid,
                                      seg_mask_paths=seg_masks if has_seg else None,
                                      subjects_dir=subjects_dir)

        result = {
            "subject_id": sid,
            "n_seg_masks": len(seg_masks),
            "n_clusters": len(clusters),
            "clusters": [{k: v for k, v in c.items() if k != "vertices"}
                         for c in clusters],
            "cluster_eval": cluster_eval,
        }
        all_results.append(result)

        with open(sdir / f"{sid}_detection.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # ── Aggregate ──
    # Cluster-level metrics (original)
    ttp = sum(r["cluster_eval"]["tp"] for r in all_results)
    tfp = sum(r["cluster_eval"]["fp"] for r in all_results)
    tfn = sum(r["cluster_eval"]["fn"] for r in all_results)
    tc = sum(r["n_clusters"] for r in all_results)
    ta = sum(r["cluster_eval"]["n_annotations"] for r in all_results)

    # MELD-style subject-level metrics
    subj_detected = sum(1 for r in all_results if r["cluster_eval"]["tp"] > 0)
    subj_missed = sum(1 for r in all_results if r["cluster_eval"]["tp"] == 0)
    subj_with_clusters = sum(1 for r in all_results if r["n_clusters"] > 0)
    n_subj = len(all_results)

    subj_sensitivity = subj_detected / max(n_subj, 1)
    subj_ppv = subj_detected / max(subj_with_clusters, 1)

    # Cluster-level
    cluster_sens = (ta - tfn) / max(ta, 1) if ta > 0 else 0
    cluster_ppv = ttp / max(tc, 1) if tc > 0 else 0

    # Mean clusters per subject (FP burden)
    mean_clusters = tc / max(n_subj, 1)
    mean_fp = tfp / max(n_subj, 1)

    logging.info(f"\n{'='*60}")
    logging.info(f"DETECTION SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"  Subjects:    {n_subj}")
    logging.info(f"  GT lesions:  {ta}")
    logging.info(f"  Clusters:    {tc} ({mean_clusters:.1f}/subject, {mean_fp:.1f} FP/subject)")
    logging.info(f"")
    logging.info(f"  --- Subject-level (MELD-style) ---")
    logging.info(f"  Detected:    {subj_detected}/{n_subj} ({subj_sensitivity:.1%})")
    logging.info(f"  Missed:      {subj_missed}/{n_subj}")
    logging.info(f"  Subject PPV: {subj_ppv:.1%} ({subj_detected}/{subj_with_clusters})")
    logging.info(f"")
    logging.info(f"  --- Cluster-level ---")
    logging.info(f"  TP={ttp}  FP={tfp}  FN={tfn}")
    logging.info(f"  Sensitivity: {cluster_sens:.1%} ({ta-tfn}/{ta})")
    logging.info(f"  Cluster PPV: {cluster_ppv:.1%} ({ttp}/{tc})")

    summary = {
        "task": "detection", "split": args.split.upper(),
        "n_subjects": n_subj, "n_annotations": ta, "n_clusters": tc,
        # Cluster-level (original)
        "tp": ttp, "fp": tfp, "fn": tfn,
        "sensitivity": float(cluster_sens), "ppv": float(cluster_ppv),
        # MELD-style subject-level
        "subject_detected": subj_detected,
        "subject_missed": subj_missed,
        "subject_sensitivity": float(subj_sensitivity),
        "subject_ppv": float(subj_ppv),
        "mean_clusters_per_subject": float(mean_clusters),
        "mean_fp_per_subject": float(mean_fp),
        "subjects": all_results,
    }
    rp = out_dir / "detection_results.json"
    with open(rp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"  Saved: {rp}")


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION: All MRIs → lesional vs healthy (AUROC etc.)
# ═══════════════════════════════════════════════════════════════

def _compute_subject_score(hemi_scores, region_baselines, atlas_per_hemi,
                            h_mean, h_std, z_clip=100.0):
    """Combine both hemispheres into a single subject-level anomaly summary.
    Skips region -1 (medial wall). Clips z-scores to [-z_clip, z_clip]."""
    all_z = []
    for hemi, scores in hemi_scores.items():
        atlas = atlas_per_hemi[hemi]; vm = ~np.isnan(scores)
        for vidx in np.where(vm)[0]:
            rid_int = int(atlas[vidx])
            if rid_int < 0:
                continue  # skip medial wall
            rid = str(rid_int); rbb = region_baselines.get(rid)
            z = ((scores[vidx] - rbb["mean"]) / rbb["std"]
                 if (rbb and rbb["std"] > 1e-6)
                 else (scores[vidx] - h_mean) / max(h_std, 1e-6))
            all_z.append(min(max(z, -z_clip), z_clip))

    if not all_z:
        return {"max_z": 0.0, "p99_z": 0.0, "p95_z": 0.0,
                "mean_top50_z": 0.0, "mfean_top100_z": 0.0,
                "mean_top500_z": 0.0,
                "n_above_2sigma": 0, "n_above_3sigma": 0}

    z = np.array(all_z)
    return {
        "max_z": float(z.max()),
        "p99_z": float(np.percentile(z, 99)),
        "p95_z": float(np.percentile(z, 95)),
        "mean_top50_z": float(np.sort(z)[-50:].mean()),
        "mean_top100_z": float(np.sort(z)[-100:].mean()),
        "mean_top500_z": float(np.sort(z)[-500:].mean()),
        "n_above_2sigma": int((z > 2).sum()),
        "n_above_3sigma": int((z > 3).sum()),
        "n_scored": len(z),
    }


def run_classification(args, dm):
    """
    Score ALL subjects (healthy + lesional). Compute subject-level score
    from both hemispheres combined. Binary classification with AUROC etc.

    Two scoring approaches:
      1. Vertex-based: top-K z-scored vertices (original)
      2. Cluster-based: run detection clustering, use top cluster score
         (more robust — healthy brains rarely form coherent high-z clusters)
    """
    device = torch.device(args.device)
    eval_dataset = getattr(args, "eval_dataset", "fcdbonn")

    # Each dataset gets its own subdirectory so runs don't overwrite each
    # other's thresholds / results.
    out_dir = pathlib.Path(args.val_output_dir) / "classification" / eval_dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline(args)
    h_mean, h_std = baseline["global_mean"], baseline["global_std"]
    rb = baseline["region_baselines"]

    fp_regions = set(_load_fp_regions(args)) | set(getattr(args, 'exclude_regions', []))

    norm_stats = dm.val_dataset.get_norm_stats()
    model = _load_model(args, device)
    _prepare_meshes(dm)

    eval_ds = dm.val_dataset if args.split == "val" else dm.test_dataset

    # Group by subject
    subj_samples = defaultdict(list)
    for s in eval_ds.samples:
        subj_samples[s["subject_id"]].append(s)

    n_patho = sum(1 for sl in subj_samples.values()
                  if any(s["has_anomaly"] for s in sl))
    n_healthy = len(subj_samples) - n_patho

    logging.info(f"\n{'='*60}")
    logging.info(f"CLASSIFICATION — {len(subj_samples)} subjects "
                 f"({n_healthy} healthy, {n_patho} patho) [{eval_dataset}]")
    logging.info(f"{'='*60}")

    all_results = []

    for si, (sid, samples) in enumerate(subj_samples.items()):
        has_anomaly = any(s["has_anomaly"] for s in samples)
        logging.info(f"  [{si+1}/{len(subj_samples)}] {sid} "
                     f"({'PATHO' if has_anomaly else 'HEALTHY'})")

        hs = {}; ha_per = {}
        hemi_data = {}

        for sample in samples:
            hemi = sample["hemi"]; mesh = dm.mesh_cache[hemi]
            r = score_hemisphere(
                model, sample["features"], sample["atlas_labels"],
                dm.mesh_cache[hemi], norm_stats, device,
                patch_size=args.patch_size, num_patches=args.num_patches,
                num_atlas_regions=args.num_atlas_regions,
            )
            hs[hemi] = r["mean"]
            ha_per[hemi] = sample["atlas_labels"].numpy()

            # Also compute z-map for cluster-based scoring
            atlas_np = sample["atlas_labels"].numpy()
            z_map = _compute_z_region_map(r["mean"], atlas_np, rb, h_mean, h_std)
            pos_np = mesh["positions"].numpy()

            hemi_data[hemi] = {
                "z_map": z_map, "positions": pos_np,
                "atlas_labels": atlas_np,
                "edge_index": mesh["edge_index"],
            }

        # Vertex-based score (original)
        ss = _compute_subject_score(hs, rb, ha_per, h_mean, h_std)

        # Cluster-based score: merge hemis, find clusters, extract features
        merged_z, merged_pos, merged_edges_list, merged_atlas = [], [], [], []
        off = 0
        for hemi in sorted(hemi_data.keys()):
            hd = hemi_data[hemi]
            n = len(hd["z_map"])
            merged_z.append(hd["z_map"])
            merged_pos.append(hd["positions"])
            merged_atlas.append(hd["atlas_labels"])
            ei = hd["edge_index"].clone(); ei += off
            merged_edges_list.append(ei)
            off += n

        merged_z = np.concatenate(merged_z)
        merged_pos = np.concatenate(merged_pos)
        merged_atlas = np.concatenate(merged_atlas)
        merged_edges = torch.cat(merged_edges_list, dim=1)

        clusters = find_clusters(
            merged_z, merged_edges, merged_pos,
            threshold=args.cluster_z_threshold,
            min_vertices=args.min_cluster_vertices,
            max_vertices=getattr(args, "max_cluster_vertices", 100000),
            max_clusters=args.max_clusters,
            atlas_labels=merged_atlas,
            sort_by=getattr(args, "cluster_sort", "area_mean_z"),
            exclude_regions=fp_regions,
        )

        # Extract cluster-based features
        if clusters:
            top = clusters[0]
            ss["top_cluster_mean_z"] = float(top["mean_z"])
            ss["top_cluster_peak_z"] = float(top["peak_z"])
            ss["top_cluster_area_z"] = float(top["area_cm2"] * top["mean_z"])
            ss["n_clusters"] = len(clusters)
            ss["sum_cluster_area"] = float(sum(c["area_cm2"] for c in clusters))
        else:
            ss["top_cluster_mean_z"] = 0.0
            ss["top_cluster_peak_z"] = 0.0
            ss["top_cluster_area_z"] = 0.0
            ss["n_clusters"] = 0
            ss["sum_cluster_area"] = 0.0

        logging.info(f"    top50={ss['mean_top50_z']:.2f}, "
                     f"clusters={ss['n_clusters']}, "
                     f"top_cl_z={ss['top_cluster_mean_z']:.2f}, "
                     f"top_cl_area_z={ss['top_cluster_area_z']:.2f}")

        all_results.append({
            "subject_id": sid, "has_anomaly": has_anomaly,
            "befund": samples[0].get("befund", ""),
            "subject_score": ss,
        })

    # ── Compute metrics ──
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.array([1 if r["has_anomaly"] else 0 for r in all_results])

    # ── Load thresholds based on --threshold_split flag ──
    val_thresholds = None
    val_primary_feature = None
    threshold_source = "youden_j"  # default: optimize on current split

    ts = getattr(args, "threshold_split", "self")

    # Explicit loading: --threshold_split val|test points to file location
    if ts != "self" and ts != args.split:
        # Load thresholds from the other split's file
        thr_path = out_dir / f"classification_thresholds_{ts}.json"
        # Fallback to legacy filename
        if not thr_path.exists():
            thr_path = out_dir / "classification_thresholds.json"
        if thr_path.exists():
            with open(thr_path) as f:
                thr_data = json.load(f)
            val_thresholds = thr_data.get("thresholds", {})
            val_primary_feature = thr_data.get("primary_feature")
            threshold_source = ts
            logging.info(f"  Loaded {ts.upper()} thresholds from: {thr_path}")
            logging.info(f"  {ts.upper()} primary feature: {val_primary_feature}")
        else:
            logging.warning(f"  No {ts} thresholds found at {thr_path}")
            logging.warning(f"  Run --mode classify --split {ts} first to establish thresholds.")
            logging.warning(f"  Falling back to Youden's J on {args.split} (WILL OVERFIT).")

    if len(np.unique(labels)) < 2:
        logging.warning("  Only one class — cannot compute AUROC")
        metrics = {"auroc": None, "n_pos": int(labels.sum()),
                   "n_neg": int((1 - labels).sum()),
                   "threshold_source": threshold_source}
    else:
        per_feat = {}
        all_feats = ["p99_z", "p95_z", "mean_top50_z",
                     "mean_top100_z", "mean_top500_z",
                     "n_above_2sigma", "n_above_3sigma",
                     "top_cluster_mean_z", "top_cluster_peak_z",
                     "top_cluster_area_z", "n_clusters", "sum_cluster_area"]
        for feat in all_feats:
            scores_arr = np.array([r["subject_score"][feat] for r in all_results])

            # AUROC is always computed fresh (threshold-independent)
            auroc = roc_auc_score(labels, scores_arr)

            # Determine operating-point threshold
            if val_thresholds and feat in val_thresholds:
                # Test mode: use fixed threshold from val
                ot = float(val_thresholds[feat])
            else:
                # Val mode (or fallback): optimize via Youden's J
                fpr, tpr, thr = roc_curve(labels, scores_arr)
                oi = np.argmax(tpr - fpr)
                ot = float(thr[oi])

            preds = (scores_arr >= ot).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            per_feat[feat] = {
                "auroc": float(auroc),
                "accuracy": float((tp + tn) / max(len(labels), 1)),
                "sensitivity": float(tp / max(tp + fn, 1)),
                "specificity": float(tn / max(tn + fp, 1)),
                "ppv": float(tp / max(tp + fp, 1)),
                "threshold": ot,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }

        # Primary feature is fixed to sum_cluster_area (the score reported in the paper).
        best_feat = "sum_cluster_area" if "sum_cluster_area" in per_feat else max(per_feat.keys(), key=lambda k: per_feat[k]["auroc"])
        # Pick best feature as primary
        # if val_primary_feature and val_primary_feature in per_feat:
        #     # Test mode: use the same primary feature as val
        #     best_feat = val_primary_feature
        # else:
        #     # Val mode: pick by best AUROC
        #     best_feat = max(per_feat.keys(), key=lambda k: per_feat[k]["auroc"])

        metrics = per_feat[best_feat].copy()
        metrics["primary_feature"] = best_feat
        metrics["per_feature"] = per_feat
        metrics["n_pos"] = int(labels.sum())
        metrics["n_neg"] = int((1 - labels).sum())
        metrics["threshold_source"] = threshold_source

        # ── Save thresholds when tuning on self ──
        if ts == "self":
            thr_to_save = {feat: fm["threshold"] for feat, fm in per_feat.items()}
            thr_save_data = {
                "primary_feature": best_feat,
                "thresholds": thr_to_save,
                "eval_dataset": eval_dataset,
                "split": args.split,
                "note": f"Thresholds optimized via Youden's J on {args.split.upper()} split. "
                        f"Load with --threshold_split {args.split}.",
            }
            thr_save_path = out_dir / f"classification_thresholds_{args.split}.json"
            with open(thr_save_path, "w") as f:
                json.dump(thr_save_data, f, indent=2)
            logging.info(f"\n  Saved {args.split} thresholds: {thr_save_path}")

    logging.info(f"\n{'='*60}")
    logging.info(f"CLASSIFICATION RESULTS ({args.split.upper()}, "
                 f"thresholds: {threshold_source})")
    logging.info(f"{'='*60}")
    if metrics.get("auroc"):
        pf = metrics["primary_feature"]
        logging.info(f"  Best feature: {pf}")
        logging.info(f"    AUROC:       {metrics['auroc']:.3f}")
        logging.info(f"    Accuracy:    {metrics['accuracy']:.1%}")
        logging.info(f"    Sensitivity: {metrics['sensitivity']:.1%} "
                     f"({metrics['tp']}/{metrics['tp']+metrics['fn']})")
        logging.info(f"    Specificity: {metrics['specificity']:.1%} "
                     f"({metrics['tn']}/{metrics['tn']+metrics['fp']})")
        logging.info(f"    Threshold:   {metrics['threshold']:.2f} "
                     f"(from {threshold_source})")
        logging.info(f"\n  All features:")
        # Sort by AUROC descending
        sorted_feats = sorted(metrics.get("per_feature", {}).items(),
                              key=lambda x: x[1]["auroc"], reverse=True)
        for feat, fm in sorted_feats:
            marker = " ←" if feat == pf else ""
            thr_tag = f" [{threshold_source}]" if threshold_source != "youden_j" else ""
            logging.info(f"    {feat:24s}: AUROC={fm['auroc']:.3f}, "
                         f"sens={fm['sensitivity']:.1%}, "
                         f"spec={fm['specificity']:.1%}, "
                         f"thr={fm['threshold']:.2f}{thr_tag}{marker}")
    else:
        logging.info(f"  Cannot compute: only {metrics.get('n_pos',0)} pos, "
                     f"{metrics.get('n_neg',0)} neg")

    summary = {
        "task": "classification", "split": args.split.upper(),
        "eval_dataset": eval_dataset,
        "threshold_source": threshold_source,
        "n_subjects": len(all_results), "metrics": metrics,
        "subjects": all_results,
    }
    rp = out_dir / "classification_results.json"
    with open(rp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"  Saved: {rp}")
