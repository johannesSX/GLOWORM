"""
visualize_results.py — Glass-brain result visualization for GLOWORM.

  --vis_result_mode glassbrain     Per-subject nilearn glass-brain in MNI
                                   space: the segmentation-mask ground-truth
                                   region (blue outline) plus the detector's
                                   top clusters (green = TP, magenta = FP),
                                   over the projected anomaly z-map. One PNG
                                   per subject.

Driven by the aggregate detection_results.json written by `--mode detect`;
point --val_output_dir at that run's directory.
"""

import json
import logging
import os
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

from dataset import N_FSAVERAGE_VERTICES


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# IDEAS pathology categories (matched against seg_mask filename substrings)
IDEAS_PATHOLOGY_PATTERNS = [
    ("fcd_type2", "FCD Type II"),
    ("fcd_type1", "FCD Type I"),
    ("dnt",       "DNT"),
    ("dnet",      "DNT"),
    ("cav",       "Cavernoma"),
    ("gliosis",   "Gliosis"),
    ("ahs",       "AHS"),
]

# Plot colors
PATHOLOGY_COLORS = {
    "FCD":             "#e41a1c",
    "FCD Type II":     "#e41a1c",
    "FCD Type I":      "#f88379",
    "DNT":             "#984ea3",
    "DNET":            "#984ea3",
    "Cavernoma":       "#ff7f00",
    "Gliosis":         "#a65628",
    "Glioma":          "#8c564b",
    "Dual pathology":  "#bcbd22",
    "AHS":             "#f781bf",
    "Other":           "#cccccc",
}


# ═══════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════════════

def get_pathology_label(sample) -> Optional[str]:
    """Extract a human-readable pathology label from a sample dict.
    Returns None for healthy subjects."""
    # Pre-resolved label (e.g. IDEAS pathology from metadata CSV).
    pre = sample.get("path_label")
    if pre:
        return pre

    seg_masks = sample.get("seg_masks", [])
    if not seg_masks:
        return None  # healthy

    # IDEAS / FCD Bonn: infer pathology from seg_mask filename patterns.
    names = " ".join(str(s).lower() for s in seg_masks)
    for pattern, label in IDEAS_PATHOLOGY_PATTERNS:
        if pattern in names:
            return label
    # FCD Bonn default
    return "FCD Type II"


# ═══════════════════════════════════════════════════════════════════════
#  Shared detection-output helpers (zmap dir / colorbar)
# ═══════════════════════════════════════════════════════════════════════

def _find_zmap_dir(val_output_dir: pathlib.Path, sid: str) -> Optional[pathlib.Path]:
    """Find the directory where a subject's zmap files live."""
    candidates = [
        val_output_dir / sid,
        val_output_dir / "detection" / sid,
        val_output_dir / "ensemble" / sid,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*_zmap.npy")):
            return c
    return None


def _find_cluster_json(zmap_dir: pathlib.Path, sid: str) -> Optional[pathlib.Path]:
    """Find the per-subject cluster JSON file."""
    candidates = [
        zmap_dir / f"{sid}_detection.json",
        zmap_dir / f"{sid}_ensemble.json",
        zmap_dir / f"{sid}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _save_standalone_colorbar(out_dir, z_thr, vmax, cmap_name,
                              label="Anomaly score"):
    """Save a standalone vertical colorbar (png/svg/pdf) for manual placement
    in the paper. Matches the heatmap range [z_thr, vmax] and colormap, so it
    is valid for every per-subject figure (which carry no colorbar)."""
    try:
        import matplotlib.pyplot as _plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cb_fig, cb_ax = _plt.subplots(figsize=(1.0, 4.2))
        sm = ScalarMappable(norm=Normalize(vmin=z_thr, vmax=vmax), cmap=cmap_name)
        sm.set_array([])
        cbar = cb_fig.colorbar(sm, cax=cb_ax)
        cbar.set_label(label, fontsize=12, rotation=270, labelpad=18)
        cbar.ax.tick_params(labelsize=10)
        for ext in ("png", "svg", "pdf"):
            cb_fig.savefig(str(pathlib.Path(out_dir) / f"anomaly_colorbar.{ext}"),
                           dpi=200, bbox_inches="tight")
        _plt.close(cb_fig)
        print(f"  Saved standalone colorbar → {out_dir}/anomaly_colorbar.(png|svg|pdf)")
    except Exception as e:
        print(f"  standalone colorbar save failed: {e}")


def _add_anomaly_colorbar(fig, z_thr, vmax, cmap="YlOrRd",
                          label="anomaly z-score"):
    """Attach a thin colorbar to a glass-brain figure, ticks at min and max only."""
    import matplotlib.cm as _cm
    import matplotlib.colors as _colors
    norm = _colors.Normalize(vmin=float(z_thr), vmax=float(vmax))
    sm = _cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Span the full vertical extent of the existing glass-brain panels.
    boxes = [a.get_position() for a in fig.axes]
    if boxes:
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
    else:
        y0, y1 = 0.12, 0.88
    # [left, bottom, width, height] in figure coords
    cax = fig.add_axes([1.02, y0, 0.012, y1 - y0])
    cbar = fig.colorbar(sm, cax=cax, ticks=[float(z_thr), float(vmax)])
    cbar.ax.set_yticklabels([f"{z_thr:.0f}", rf"$\geq${vmax:.0f}"])
    cbar.set_label(label, fontsize=9, rotation=270, labelpad=0)
    cbar.ax.tick_params(labelsize=8, pad=2)
    return cbar


# ═══════════════════════════════════════════════════════════════════════
#  Mode 4: Per-Subject Glass Brain (GT region + detected clusters)
# ═══════════════════════════════════════════════════════════════════════

# MNI152 2mm voxel grid
MNI152_SHAPE = (91, 109, 91)
MNI152_AFFINE = np.array([
    [-2.0,  0.0,  0.0,  90.0],
    [ 0.0,  2.0,  0.0, -126.0],
    [ 0.0,  0.0,  2.0,  -72.0],
    [ 0.0,  0.0,  0.0,    1.0],
])


def _mni_to_voxel(xyz):
    inv = np.linalg.inv(MNI152_AFFINE)
    coords = np.asarray(xyz, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords[None]
    homog = np.column_stack([coords, np.ones(len(coords))])
    return np.round((inv @ homog.T).T[:, :3]).astype(np.int64)


def _accumulate_to_volume(positions, sigma_vox: float = 2.0):
    vol = np.zeros(MNI152_SHAPE, dtype=np.float32)
    if not len(positions):
        return vol
    vox = _mni_to_voxel(positions)
    valid = ((vox[:, 0] >= 0) & (vox[:, 0] < MNI152_SHAPE[0]) &
             (vox[:, 1] >= 0) & (vox[:, 1] < MNI152_SHAPE[1]) &
             (vox[:, 2] >= 0) & (vox[:, 2] < MNI152_SHAPE[2]))
    vox = vox[valid]
    for v in vox:
        vol[v[0], v[1], v[2]] += 1.0
    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=sigma_vox)
    except ImportError:
        pass
    return vol


def _clusters_to_volume(clusters, sigma_vox: float = 1.0):
    """Rasterize clusters as filled spheres into the MNI152 grid.

    The aggregate detection JSON keeps each cluster's `centroid` and
    `area_cm2` but not its vertex set, so we approximate each cluster as a
    sphere whose radius grows with its area. Returns a (91,109,91) volume or
    None if there is nothing to draw.
    """
    pts = []
    for cl in clusters:
        cent = cl.get("centroid")
        if cent is None:
            continue
        area = float(cl.get("area_cm2", 0.0) or 0.0)
        # crude equivalent radius (mm) from surface area; clamped to a sane band
        radius = float(np.clip(np.sqrt(max(area, 1.0)) * 2.0, 4.0, 16.0))
        step = 2.0
        rng = np.arange(-radius, radius + step, step)
        dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
        m = (dx * dx + dy * dy + dz * dz) <= radius * radius
        offs = np.stack([dx[m], dy[m], dz[m]], axis=1)
        pts.append(np.asarray(cent, dtype=np.float64)[None, :] + offs)
    if not pts:
        return None
    return _accumulate_to_volume(np.vstack(pts).tolist(), sigma_vox=sigma_vox)


def _zmap_to_volume(zmaps, dm, z_thr: float = 2.0, sigma_vox: float = 1.0):
    """Project a per-vertex anomaly z-map onto the MNI152 grid.

    zmaps:  {hemi: (V,) array of z-scores} loaded from {hemi}_zmap.npy
    Uses dm.mesh_cache[hemi]["positions"] (full-res fsaverage mesh, MNI mm) to
    place each supra-threshold vertex, keeping the max |z| per voxel.
    Returns a (91,109,91) volume, or None if nothing is above threshold.
    """
    vol = np.zeros(MNI152_SHAPE, dtype=np.float32)
    any_pts = False
    for hemi, z in zmaps.items():
        if hemi not in dm.mesh_cache:
            continue
        pos = dm.mesh_cache[hemi]["positions"].numpy()
        z = np.asarray(z, dtype=np.float64).ravel()
        n = min(len(z), len(pos))
        z, pos = z[:n], pos[:n]
        z = np.where(np.isnan(z), 0.0, z)
        mask = np.abs(z) > z_thr
        if not np.any(mask):
            continue
        vox = _mni_to_voxel(pos[mask])
        zz = np.abs(z[mask]).astype(np.float32)
        valid = ((vox[:, 0] >= 0) & (vox[:, 0] < MNI152_SHAPE[0]) &
                 (vox[:, 1] >= 0) & (vox[:, 1] < MNI152_SHAPE[1]) &
                 (vox[:, 2] >= 0) & (vox[:, 2] < MNI152_SHAPE[2]))
        vox, zz = vox[valid], zz[valid]
        for (i, j, k), zval in zip(vox, zz):
            if zval > vol[i, j, k]:
                vol[i, j, k] = zval
            any_pts = True
    if not any_pts:
        return None
    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=sigma_vox)
    except ImportError:
        pass
    return vol


def _points_to_filled_volume(points, close_iter: int = 2, dilate_iter: int = 0,
                             sigma_vox: float = 1.0):
    """Rasterize points into a *contiguous* filled blob in the MNI152 grid.

    The cortex is a thin folded sheet, so raw surface vertices land as
    scattered voxels with gaps. We mark occupied voxels, then morphologically
    close (bridge inter-vertex sampling gaps without inflating the outer
    boundary) and fill enclosed holes, so a connected surface cluster becomes
    one solid region; then lightly smooth. `dilate_iter` is off by default
    because dilation enlarges the region beyond its true extent. Returns a
    volume or None.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return None
    vox = _mni_to_voxel(points)
    valid = ((vox[:, 0] >= 0) & (vox[:, 0] < MNI152_SHAPE[0]) &
             (vox[:, 1] >= 0) & (vox[:, 1] < MNI152_SHAPE[1]) &
             (vox[:, 2] >= 0) & (vox[:, 2] < MNI152_SHAPE[2]))
    vox = vox[valid]
    if len(vox) == 0:
        return None
    mask = np.zeros(MNI152_SHAPE, dtype=bool)
    mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    try:
        from scipy.ndimage import (binary_closing, binary_dilation,
                                   binary_fill_holes, gaussian_filter)
        if close_iter > 0:
            mask = binary_closing(mask, iterations=close_iter)
        if dilate_iter > 0:
            mask = binary_dilation(mask, iterations=dilate_iter)
        mask = binary_fill_holes(mask)
        vol = gaussian_filter(mask.astype(np.float32), sigma=sigma_vox)
    except ImportError:
        vol = mask.astype(np.float32)
    # Normalize to peak 1.0 so a fixed 0.5 contour renders for ANY cluster
    # size (Gaussian smoothing otherwise drops small blobs well below 0.5).
    m = float(vol.max())
    if m > 0:
        vol = vol / m
    return vol


def _cluster_shape_volumes(zmaps, dm, clusters, z_thr: float = 2.0,
                           max_assign_mm: float = 25.0, sigma_vox: float = 0.8):
    """Reconstruct TP / FP cluster *extents* from the z-map.

    The aggregate JSON has no vertex sets, but the detected clusters are the
    supra-threshold regions of the z-map. So: take supra-threshold vertices,
    assign each to the nearest detected-cluster centroid (within
    max_assign_mm), and inherit that cluster's is_tp label. Returns
    (tp_vol, fp_vol); either may be None.
    """
    cents, is_tp = [], []
    for cl in clusters:
        c = cl.get("centroid")
        if c is None:
            continue
        cents.append(c)
        is_tp.append(bool(cl.get("is_tp", False)))
    if not cents:
        return None, None
    cents = np.asarray(cents, dtype=np.float64)
    is_tp = np.asarray(is_tp, dtype=bool)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cents)
    except Exception:
        tree = None

    tp_pts, fp_pts = [], []
    for hemi, z in zmaps.items():
        if hemi not in dm.mesh_cache:
            continue
        pos = dm.mesh_cache[hemi]["positions"].numpy()
        z = np.abs(np.where(np.isnan(z), 0.0,
                            np.asarray(z, dtype=np.float64).ravel()))
        n = min(len(z), len(pos))
        z, pos = z[:n], pos[:n]
        sel = pos[z > z_thr]
        if len(sel) == 0:
            continue
        if tree is not None:
            dist, idx = tree.query(sel, k=1)
        else:
            d = np.linalg.norm(sel[:, None, :] - cents[None, :, :], axis=2)
            idx = d.argmin(axis=1)
            dist = d[np.arange(len(sel)), idx]
        keep = dist <= max_assign_mm
        sel, idx = sel[keep], idx[keep]
        tpm = is_tp[idx]
        if np.any(tpm):
            tp_pts.append(sel[tpm])
        if np.any(~tpm):
            fp_pts.append(sel[~tpm])

    tp_vol = (_points_to_filled_volume(np.vstack(tp_pts))
              if tp_pts else None)
    fp_vol = (_points_to_filled_volume(np.vstack(fp_pts))
              if fp_pts else None)
    return tp_vol, fp_vol


def _top_cluster_components(zmaps, dm, clusters, z_thr: float = 2.0,
                           top_k: int = 3):
    """Recover each top-k cluster's REAL connected component on the mesh graph.

    The detector forms a cluster as a connected component of supra-threshold
    vertices over the mesh edges. We reproduce that exactly: for each of the
    top-k clusters (by rank), flood-fill from its `peak_vertex` across
    supra-threshold vertices using `dm.mesh_cache[hemi]["edge_index"]`. The
    component IS the cluster (one coherent region, not fragments). Returns a
    list of dicts: positions, is_tp, rank, hemi, n_found, n_json.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        return []

    cl = [c for c in clusters if c.get("peak_vertex") is not None
          and c.get("hemi") in zmaps and c.get("hemi") in dm.mesh_cache]
    cl = sorted(cl, key=lambda c: c.get("rank", 10**9))[:top_k]
    if not cl:
        return []

    # Per-hemi connected-component labels over the supra-threshold subgraph
    # (computed once per hemi, then reused for every cluster in that hemi).
    # NOTE: detection clusters on a MERGED lh+rh graph, so peak_vertex is a
    # merged/global index (rh shifted by the lh vertex count). Convert it back
    # to a per-hemi index before indexing the per-hemi z-map / labels.
    n_lh = (len(np.asarray(zmaps["lh"]).ravel()) if "lh" in zmaps else 163842)
    cache = {}
    out = []
    for c in cl:
        hemi = c["hemi"]
        pv = int(c["peak_vertex"]) - (n_lh if hemi == "rh" else 0)
        if hemi not in cache:
            z = np.asarray(zmaps[hemi], dtype=np.float64).ravel()
            z = np.abs(np.where(np.isnan(z), 0.0, z))
            ei = dm.mesh_cache[hemi]["edge_index"]
            ei = ei.numpy() if hasattr(ei, "numpy") else np.asarray(ei)
            pos = dm.mesh_cache[hemi]["positions"]
            pos = pos.numpy() if hasattr(pos, "numpy") else np.asarray(pos)
            N = len(z)
            supra = z > z_thr
            keep_e = supra[ei[0]] & supra[ei[1]]
            e0, e1 = ei[0][keep_e], ei[1][keep_e]
            A = coo_matrix((np.ones(len(e0), dtype=np.uint8), (e0, e1)),
                           shape=(N, N))
            _, labels = connected_components(A, directed=False)
            cache[hemi] = (labels, supra, pos)
        labels, supra, pos = cache[hemi]
        if pv < 0 or pv >= len(labels) or not supra[pv]:
            # Peak isn't supra-threshold at this z_thr -> can't grow a region.
            # Still return the cluster (centroid fallback) so all top-k show.
            out.append({
                "positions": None,
                "centroid": c.get("centroid"),
                "is_tp": bool(c.get("is_tp", False)),
                "rank": c.get("rank"),
                "hemi": hemi,
                "n_found": 0,
                "n_json": int(c.get("n_vertices", -1)),
            })
            continue
        comp = np.where((labels == labels[pv]) & supra)[0]
        out.append({
            "positions": pos[comp],
            "centroid": c.get("centroid"),
            "is_tp": bool(c.get("is_tp", False)),
            "rank": c.get("rank"),
            "hemi": hemi,
            "n_found": int(len(comp)),
            "n_json": int(c.get("n_vertices", -1)),
        })
    return out


def _load_detection_records(val_output_dir: pathlib.Path) -> Dict[str, dict]:
    """Load the aggregate detection JSON -> {subject_id: record}.

    `--mode detect` writes `detection_results.json` under the `detection/`
    subfolder of --val_output_dir, with a top-level `subjects` list. Each record
    carries per-cluster `centroid` + `is_tp` and `cluster_eval.gt_mni_centroid`,
    all in MNI mm. The root of --val_output_dir is also checked for backward
    compatibility. Returns {} if nothing usable is found.
    """
    candidates = [
        val_output_dir / "detection" / "detection_results.json",
        val_output_dir / "detection_results.json",
        val_output_dir / "ensemble" / "detection_results.json",
    ]
    candidates += sorted(val_output_dir.glob("*detection*result*.json"))
    candidates += sorted(val_output_dir.glob("detection*/detection_results.json"))
    seen = set()
    for c in candidates:
        if c in seen or not c.exists():
            continue
        seen.add(c)
        try:
            with open(c) as f:
                data = json.load(f)
        except Exception:
            continue
        subjects = data.get("subjects", []) if isinstance(data, dict) else []
        recs = {r["subject_id"]: r for r in subjects if isinstance(r, dict)
                and "subject_id" in r}
        if recs:
            print(f"  Loaded {len(recs)} subject records from {c}")
            return recs
    return {}


def run_glassbrain(args, dm):
    """Per-subject nilearn glass-brain: GT lesion region + detected clusters.

    Driven by the aggregate detection_results.json under --val_output_dir,
    which carries each subject's cluster centroids (+is_tp) and GT MNI centroid.
    The GT is drawn as the full seg-mask region in MNI when available, else as
    the GT centroid point. Markers: blue = TP, orange = FP. One figure per
    subject. Restrict with --vis_subject "sid1 sid2".
    """
    try:
        from nilearn import plotting as nl_plotting
        import nibabel as nib
    except ImportError:
        print("  nilearn or nibabel not installed. Run: pip install nilearn nibabel")
        return
    try:
        from validation import compute_seg_mask_mni_points
    except ImportError:
        compute_seg_mask_mni_points = None

    out_dir = pathlib.Path(args.val_output_dir) / "viz_results" / "glassbrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GLASS BRAIN — per Subject (GT region + detected clusters)")
    print(f"{'='*60}")

    val_output_dir = pathlib.Path(args.val_output_dir)
    records = _load_detection_records(val_output_dir)
    if not records:
        print(f"  No detection_results.json with a 'subjects' list found under "
              f"{val_output_dir}.\n  Run `--mode detect` first and point "
              f"--val_output_dir at that directory (e.g. .../detection_3/).")
        return

    # subject_id -> sample dict (for seg-mask paths + subjects_dir).
    # Search ALL splits so a JSON subject always resolves to its seg-masks,
    # even if the detection run used a different split than --split here.
    sample_by_sid: Dict[str, dict] = {}
    for ds in (getattr(dm, "train_dataset", None),
               getattr(dm, "val_dataset", None),
               getattr(dm, "test_dataset", None)):
        if ds is None:
            continue
        for s in getattr(ds, "samples", []):
            sample_by_sid.setdefault(s["subject_id"], s)

    target_sids = None
    if getattr(args, "vis_subject", None):
        target_sids = set(args.vis_subject.split())

    rendered = 0
    for sid, rec in sorted(records.items()):
        if target_sids is not None and sid not in target_sids:
            continue

        # ── Clusters -> TP / FP (full dicts, for centroid + area) ──
        tp_clusters, fp_clusters = [], []
        for cl in rec.get("clusters", []):
            if cl.get("centroid") is None:
                continue
            if bool(cl.get("is_tp", False)):
                tp_clusters.append(cl)
            else:
                fp_clusters.append(cl)

        # ── GT region: full seg-mask in MNI, else the GT centroid point ──
        gt_positions = []
        gt_reason = ""
        sample = sample_by_sid.get(sid)
        seg_masks = (sample or {}).get("seg_masks", [])
        subjects_dir = (sample or {}).get("subjects_dir",
                                          getattr(dm, "subjects_dir", None))
        if sample is None:
            gt_reason = "no sample for sid in any split"
        elif not seg_masks:
            gt_reason = "sample has no seg_masks"
        elif compute_seg_mask_mni_points is None:
            gt_reason = "compute_seg_mask_mni_points unavailable"
        else:
            try:
                mni_pts, nvox = compute_seg_mask_mni_points(
                    seg_masks, sid, subjects_dir)
                if mni_pts is not None and len(mni_pts):
                    gt_positions = np.asarray(mni_pts, dtype=np.float64).tolist()
                else:
                    gt_reason = f"seg-mask projection returned 0 pts (nvox={nvox})"
            except Exception as e:
                gt_reason = f"seg-mask projection error: {e}"
        if not gt_positions:
            ce = rec.get("cluster_eval", {}) or {}
            gtc = ce.get("gt_mni_centroid")
            if gtc is not None:
                gt_positions = [gtc]
            if gt_reason:
                print(f"    [GT fallback] {sid}: {gt_reason}")

        if not (gt_positions or tp_clusters or fp_clusters):
            continue

        path_label = (get_pathology_label(sample) if sample else None) or "lesion"
        n_gt, n_tp, n_fp = len(gt_positions), len(tp_clusters), len(fp_clusters)

        # ── Background: anomaly z-map projected to MNI (continuous heatmap) ──
        zmaps = {}
        zmap_dir = _find_zmap_dir(val_output_dir, sid)
        if zmap_dir is not None:
            for hemi in ("lh", "rh"):
                zp = zmap_dir / f"{hemi}_zmap.npy"
                if zp.exists():
                    try:
                        zmaps[hemi] = np.load(str(zp))
                    except Exception:
                        pass
        z_thr = float(getattr(args, "vis_z_threshold", 2.0))
        cmap_name = getattr(args, "vis_cmap", "YlOrRd")
        vmax = float(getattr(args, "vis_vmax", 8.0))
        anomaly_vol = _zmap_to_volume(zmaps, dm, z_thr=z_thr) if zmaps else None

        top_k_disp = int(getattr(args, "vis_top_clusters", 3))
        title = (f"{sid}   ·   {path_label}        "
                 f"GT (blue)    TP (green)    FP (magenta)        "
                 f"TP: {n_tp}    FP: {n_fp}")
        if anomaly_vol is not None:
            display = nl_plotting.plot_glass_brain(
                nib.Nifti1Image(anomaly_vol, MNI152_AFFINE),
                display_mode="lyrz", cmap=cmap_name, colorbar=False,
                plot_abs=True, threshold=z_thr, vmin=z_thr, vmax=vmax,
                symmetric_cbar=False)
        else:
            display = nl_plotting.plot_glass_brain(None, display_mode="lyrz")

        # Clean full-width title (no black box): black text via figure suptitle.
        try:
            import matplotlib.pyplot as _plt
            fig = _plt.gcf()
            fig.suptitle(title, fontsize=9, color="black", x=0.5, y=0.97,
                         ha="center")
        except Exception:
            try:
                display.title(title, size=9, color="k", bgcolor="none")
            except Exception:
                pass

        # No colorbar inside the per-subject figures — it's saved separately
        # once (see end of this function) so it can be placed manually in the
        # paper. vmin=z_thr above keeps the heatmap on the full [z_thr, vmax]
        # range, matching that standalone colorbar.

        # ── Extra layers: TOP-3 clusters as their REAL connected components ──
        # Each top-k cluster (by rank) is recovered as the connected component
        # of supra-threshold vertices around its peak_vertex on the mesh graph
        # -> one coherent region per cluster, faithful to what the detector
        # found. Outlined by TP/FP; falls back to lumped scatter if no z-map.
        top_k = int(getattr(args, "vis_top_clusters", 3))
        cluster_z = float(getattr(args, "cluster_z_threshold", 2.5))
        comps = (_top_cluster_components(zmaps, dm, rec.get("clusters", []),
                                         z_thr=cluster_z, top_k=top_k)
                 if zmaps else [])
        if comps:
            for comp in comps:
                color = "#2ca02c" if comp["is_tp"] else "#ff00ff"  # green / magenta
                pts = comp.get("positions")
                if pts is not None and len(pts):
                    vol = _points_to_filled_volume(pts, close_iter=2,
                                                   dilate_iter=0)
                    if vol is not None and vol.max() > 0:
                        try:
                            display.add_contours(
                                nib.Nifti1Image(vol, MNI152_AFFINE),
                                levels=[0.5], colors=color, linewidths=2.2)
                        except Exception:
                            display.add_overlay(
                                nib.Nifti1Image(vol, MNI152_AFFINE),
                                cmap="Greens" if comp["is_tp"] else "spring",
                                threshold=0.5)
                        continue
                # fallback: peak below threshold / no region -> centroid marker
                if comp.get("centroid") is not None:
                    display.add_markers(
                        np.asarray([comp["centroid"]], dtype=np.float64),
                        marker_color=color, marker_size=45)
            sanity = "  ".join(
                f"r{c['rank']}:{c['n_found']}/{c['n_json']}v"
                f"{'(TP)' if c['is_tp'] else '(FP)'}"
                f"{'' if c.get('positions') is not None else '[mark]'}"
                for c in comps)
            print(f"    [top{top_k}] {sid}: {len(comps)} clusters  {sanity}")
        else:
            # No z-maps -> reconstruct lumped TP/FP extents (approximate)
            tp_vol = _clusters_to_volume(tp_clusters, sigma_vox=1.0)
            fp_vol = _clusters_to_volume(fp_clusters, sigma_vox=1.0)
            if tp_vol is not None and tp_vol.max() > 0:
                display.add_contours(nib.Nifti1Image(tp_vol, MNI152_AFFINE),
                                     levels=[0.5], colors="green", linewidths=2.0)
            if fp_vol is not None and fp_vol.max() > 0:
                display.add_contours(nib.Nifti1Image(fp_vol, MNI152_AFFINE),
                                     levels=[0.5], colors="black", linewidths=1.5)

        # GT mask shape: full seg-mask region as a contiguous blue outline
        if gt_positions and len(gt_positions) > 1:
            gt_vol = _points_to_filled_volume(np.asarray(gt_positions),
                                              close_iter=2, dilate_iter=0)
            if gt_vol is not None and gt_vol.max() > 0:
                gt_nii = nib.Nifti1Image(gt_vol, MNI152_AFFINE)
                try:
                    display.add_contours(gt_nii, levels=[0.5],
                                         colors="blue", linewidths=2.5)
                except Exception:
                    display.add_overlay(gt_nii, cmap="Blues", threshold=0.5)
        elif gt_positions:
            display.add_markers(np.asarray(gt_positions, dtype=np.float64),
                                marker_color="blue", marker_size=90)

        out_path = out_dir / f"{sid}_glassbrain.png"
        try:
            import matplotlib.pyplot as _plt
            _fig = _plt.gcf()
            if rendered == 0:  # one-time diagnostic
                ws = [round(a.get_position().width, 3) for a in _fig.axes]
                print(f"    [cbar debug] {len(_fig.axes)} axes, widths={ws}")
            _add_anomaly_colorbar(
                _fig,
                float(getattr(args, "vis_z_threshold", 2.0)),
                float(getattr(args, "vis_vmax", 8.0)),
                getattr(args, "vis_cmap", "YlOrRd"),
            )
            _fig.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
        except Exception:
            display.savefig(str(out_path), dpi=180)
        display.close()
        rendered += 1
        print(f"    {sid:48s} {path_label:18s} "
              f"GTpts={n_gt} TP={n_tp} FP={n_fp} → {out_path.name}")

    print(f"\n  Rendered {rendered} per-subject glass-brain figures → {out_dir}")

    # Standalone colorbar (saved once) — place it manually in the paper; the
    # per-subject glass brains deliberately carry no colorbar.
    _save_standalone_colorbar(
        out_dir,
        float(getattr(args, "vis_z_threshold", 2.0)),
        float(getattr(args, "vis_vmax", 8.0)),
        getattr(args, "vis_cmap", "YlOrRd"))


# ═══════════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════════

def run_result_visualization(args, dm=None):
    """Dispatch to the chosen --vis_result_mode sub-mode.

    Only the glass-brain visualization is shipped in the public repo.
    """
    sub_mode = getattr(args, "vis_result_mode", "glassbrain")

    if sub_mode == "glassbrain":
        if dm is None:
            raise ValueError("glassbrain mode requires loaded data (dm)")
        run_glassbrain(args, dm)
    else:
        raise ValueError(f"Unknown vis_result_mode: {sub_mode}. "
                         f"Only 'glassbrain' is available.")