"""
run_cyclevae.py — Entry point for the GLOWORM Graph Cycle-VAE
=============================================================

Cycle-consistent dual VAE for unsupervised cortical-surface anomaly detection.
  Path A: geometry(curv, sulc) → structure(thickness, area, wg_pct)
  Path B: structure → geometry
  Cycle:  geometry → structure_pred → geometry_recon (and reverse)

Anomaly = cross-modal prediction error + cycle inconsistency.
Uses morphometric features only (T1-derived; no FLAIR/T2/SWI).

Typical workflow (see README for the data layout and full details):

  # 1. Train on healthy controls (IXI / FCD Bonn healthy / combined)
  python run_gloworm.py --mode train --train_dataset ixi --eval_dataset fcdbonn

  # 2. Healthy baseline (per-region reference statistics)
  python run_gloworm.py --mode baseline --train_dataset ixi --eval_dataset fcdbonn \
      --val_checkpoint checkpoints_cyclevae/best.pt \
      --val_output_dir results/fcdbonn

  # 3. Lesion detection on the validation split
  python run_gloworm.py --mode detect --eval_dataset fcdbonn --split val \
      --val_checkpoint checkpoints_cyclevae/best.pt \
      --val_output_dir results/fcdbonn --cluster_match_distance 20

  # 4. Subject-level classification (AUROC etc.)
  python run_gloworm.py --mode classify --eval_dataset fcdbonn --split val \
      --val_checkpoint checkpoints_cyclevae/best.pt \
      --val_output_dir results/fcdbonn --threshold_split self

  # 5. Glass-brain figures for the detected subjects
  python run_gloworm.py --mode visualize_results --vis_result_mode glassbrain \
      --eval_dataset fcdbonn --split val --val_output_dir results/fcdbonn
"""

import argparse
import torch

from pl import run_train_loop
from validation import (run_healthy_baseline, run_detection, run_classification)


def main():
    parser = argparse.ArgumentParser(description="Graph Cycle-VAE")

    # ── Mode ──
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "baseline", "detect", "classify",
                                 "visualize_results"])

    # ── Model ──
    parser.add_argument("--dim_h", type=int, default=128)
    parser.add_argument("--dim_pe", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent dim (smaller for cross-modal: 32-64)")
    parser.add_argument("--num_pool_levels", type=int, default=2)
    parser.add_argument("--pool_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_layers_per_level", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--edge_weight", type=float, default=1.0,
                        help="Scale on raw edge geometry into the edge encoder. "
                             "0 = no edge features, 1 = full. Sweep e.g. 0 / 0.25 / 0.5 / 1.0.")

    # ── CycleVAE specific ──
    parser.add_argument("--num_morpho", type=int, default=5,
                        help="Number of morphometric features")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL weight (lower=better recon for anomaly detection)")
    parser.add_argument("--lambda_cycle", type=float, default=1.0,
                        help="Cycle consistency weight")
    parser.add_argument("--split_mode", type=str, default="morpho_only",
                        choices=["morpho_only", "custom"],
                        help="morpho_only: geometry(curv,sulc)↔structure(thick,area,wg_pct), "
                             "custom: use --indices_a and --indices_b")
    parser.add_argument("--indices_a", nargs="*", type=int, default=None,
                        help="Feature indices for path A (custom split_mode)")
    parser.add_argument("--indices_b", nargs="*", type=int, default=None,
                        help="Feature indices for path B (custom split_mode)")

    # ── Data paths ──
    parser.add_argument("--subjects_dir_ext", type=str,
                        default="data/fastsurfer_subjects",
                        help="FastSurfer subjects directory containing ALL "
                             "cohorts (subject folders prefixed fcdbonn__/"
                             "ideas__/ixi__), plus fsaverage/ and "
                             "fsaverage_common/.")
    parser.add_argument("--eval_dataset", type=str, default="fcdbonn",
                        choices=["fcdbonn", "idea"],
                        help="Cohort to evaluate detection/classification on: "
                             "'fcdbonn' (Bonn FCD) or 'idea' (IDEAS).")
    parser.add_argument("--train_dataset", type=str, default=None,
                        choices=["ixi", "fcdbonn", "combined"],
                        help="Healthy cohort to train on. 'ixi' = IXI healthy "
                             "controls, 'fcdbonn' = FCD Bonn healthy controls, "
                             "'combined' = FCD Bonn + IXI healthy pooled.")
    parser.add_argument("--ext_data_dir", type=str,
                        default="data/FCDBONN",
                        help="Path to the raw dataset directory holding the "
                             "segmentation masks (ground truth).")
    parser.add_argument("--ideas_filter", type=str, default="all",
                        choices=["all", "fcd", "fcd_type2", "fcd_including_dual",
                                 "dnt", "cav", "gliosis", "mcd", "hs", "gl",
                                 "all_cortical"],
                        help="IDEAS pathology filter: "
                             "'fcd'=FCD, 'fcd_type2'=IIa+IIb, 'dnt'=tumors, "
                             "'cav'=cavernomas, 'gliosis', 'mcd'=MCD/heterotopia, "
                             "'hs'=hippocampal sclerosis, 'gl'=ganglioglioma, "
                             "'all_cortical'=FCD+DNT+CAV+GL+MCD.")

    # ── Hemisphere ──
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"],
                        choices=["lh", "rh"])

    #  ── Atlas ──
    parser.add_argument("--atlas", type=str, default="dkt",
                        choices=["dkt", "destrieux", "hcp"],
                        help="Atlas for positional encoding and z-scoring "
                             "(dkt=34+2 regions, destrieux=74+2, hcp=180+1)")

    # ── Data options ──
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_after_X_epochs", type=int, default=1)
    parser.add_argument("--samples_per_epoch", type=int, default=25000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--patch_size", type=int, default=5000)

    # ── Training ──
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--val_samples_in_train", type=int, default=10000,
                        help="Random patches sampled per validation pass "
                             "(averaged → low-variance val loss). Default: one per val graph.")

    # ── Validation ──
    parser.add_argument("--baseline_dataset", type=str, default="eval",
                        choices=["eval", "train"],
                        help="Which dataset to compute baseline on: "
                             "'eval'=--eval_dataset (default), "
                             "'train'=--train_dataset")
    parser.add_argument("--baseline_split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Which split of baseline_dataset to use: "
                             "'train'=training subjects, "
                             "'val'=validation subjects (default), "
                             "'test'=test subjects")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Which split to evaluate on")
    parser.add_argument("--threshold_split", choices=["val", "test", "self"],
                        default="self",
                        help="Where classification threshold was tuned. "
                             "'self'=tune on current split (Youden's J), "
                             "'val'=load from val, 'test'=load from test.")
    parser.add_argument("--num_patches", type=int, default=200) # 200
    parser.add_argument("--num_patches_healthy", type=int, default=200)
    parser.add_argument("--max_healthy", type=int, default=1000)
    parser.add_argument("--val_output_dir", default="validation_results_cyclevae/")
    parser.add_argument("--val_checkpoint", default="checkpoints_cyclevae/best.pt")

    # ── Cluster detection ──
    parser.add_argument("--cluster_z_threshold", type=float, default=2.5) # 4.5
    parser.add_argument("--min_cluster_vertices", type=int, default=25)
    parser.add_argument("--max_cluster_vertices", type=int, default=100000)
    parser.add_argument("--max_clusters", type=int, default=3)
    parser.add_argument("--cluster_match_distance", type=float, default=20.0)
    parser.add_argument("--cluster_sort", type=str, default="area_mean_z",
                        choices=["mean_z", "area_mean_z", "peak_z"],
                        help="Cluster ranking: mean_z, area_mean_z (area*mean_z), peak_z")
    parser.add_argument("--exclude_regions", nargs="*", type=int, default=[],
                        help="Atlas region IDs to exclude from clusters (e.g. 22 24 35)")

    # ── Misc ──
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--tensorboard_dir", type=str, default="runs/cycle_vae")

    # ── Compat with GraphBetaVAE argparse (needed by build_dataset) ──
    parser.add_argument("--dim_mni", type=int, default=32)
    parser.add_argument("--dim_atlas", type=int, default=16)
    parser.add_argument("--num_atlas_regions", type=int, default=36)
    parser.add_argument("--in_edge_dim", type=int, default=4)
    parser.add_argument("--mni_sigma", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--loss_type", type=str, default="H")

    # ── Visualization (glassbrain only) ──
    parser.add_argument("--vis_subject", type=str, default=None,
                        help="Subject ID(s) to visualize (space-separated). "
                             "Default: all subjects in detection_results.json.")
    parser.add_argument("--vis_result_mode", type=str, default="glassbrain",
                        choices=["glassbrain"],
                        help="Sub-mode for --mode visualize_results (glassbrain).")
    parser.add_argument("--vis_z_threshold", type=float, default=2.0,
                        help="z-score threshold for the glassbrain anomaly map")
    parser.add_argument("--vis_cmap", type=str, default="YlOrRd",
                        help="matplotlib colormap for the surface anomaly map "
                             "(e.g. YlOrRd, hot, inferno, magma, turbo)")
    parser.add_argument("--vis_vmax", type=float, default=8.0,
                        help="Fixed upper limit of the anomaly colorbar, shared "
                             "across all subjects (keeps figures comparable)")
    parser.add_argument("--vis_top_clusters", type=int, default=3,
                        help="How many top-ranked clusters to outline in the "
                             "glassbrain (each as its real connected component)")

    args = parser.parse_args()

    from dataset import ATLAS_CONFIG
    args.num_atlas_regions = ATLAS_CONFIG[args.atlas]["num_regions"]

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # ── Data ──
    print("\n=== Loading data ===")
    from build_dataset import SurfaceDataModule

    dm = SurfaceDataModule(args)
    train_loader, val_loader, test_loader = dm.get_loaders()

    # Get num_features from whichever dataset has samples
    if len(dm.train_dataset) > 0:
        args.num_features = dm.train_dataset.num_node_features
    elif len(dm.val_dataset) > 0:
        args.num_features = dm.val_dataset.num_node_features
    elif len(dm.test_dataset) > 0:
        args.num_features = dm.test_dataset.num_node_features
    else:
        raise RuntimeError("No samples loaded in any split")

    print(f"  Features: {args.num_features} (morpho only)")
    print(f"  Split mode: {args.split_mode}")
    print(f"  Patch size: {args.patch_size}")
    sizes = [args.patch_size]
    for i in range(args.num_pool_levels):
        sizes.append(int(sizes[-1] * args.pool_ratio))
    print(f"  Pool: {' → '.join(str(s) for s in sizes)}")

    # ── Run ──
    if args.mode == "train":
        run_train_loop(args, train_loader, val_loader, device)
    elif args.mode == "baseline":
        run_healthy_baseline(args, dm)
    elif args.mode == "detect":
        run_detection(args, dm)
    elif args.mode == "classify":
        run_classification(args, dm)
    elif args.mode == "visualize_results":
        from visualize_results import run_result_visualization
        run_result_visualization(args, dm)


if __name__ == "__main__":
    main()
