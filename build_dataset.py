"""
build_dataset.py — Data module for cortical surface graph datasets
===================================================================

Scans a FastSurfer subjects directory for the external cohorts
(FCD Bonn, IDEAS, IXI), locates each subject's segmentation mask (ground
truth), splits into train / validation / test, and builds
CorticalSurfaceDataset instances + DataLoaders for each split.

Usage (from run_gloworm.py):
  from build_dataset import SurfaceDataModule
  dm = SurfaceDataModule(args)
  train_loader, val_loader, test_loader = dm.get_loaders()
"""

import argparse
import copy
import csv
import glob
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tqdm
from sklearn import model_selection as sk_model_selection
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler

from dataset import (
    CorticalSurfaceDataset,
    load_fsaverage_mesh,
    compute_edge_features,
    HEMIS,
    N_FSAVERAGE_VERTICES,
)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# SurfaceDataModule
# ─────────────────────────────────────────────────────────

class SurfaceDataModule:
    """
    Data module for cortical surface graph datasets.

    Handles:
      1. Scanning the FastSurfer subjects directory for the requested cohort
      2. Locating each subject's segmentation mask (ground truth)
      3. Splitting into train / validation / test
      4. Building CorticalSurfaceDataset + DataLoader for each split

    Args:
        args: Namespace with:
          - subjects_dir_ext:   FastSurfer subjects directory (all cohorts)
          - eval_dataset:       Cohort to evaluate on ('fcdbonn' | 'idea')
          - train_dataset:      Healthy training cohort ('fcdbonn'|'ixi'|'combined')
          - ext_data_dir:       Raw dataset dir holding the seg masks
          - ideas_filter:       Optional IDEAS pathology filter
          - hemispheres:        List of hemispheres
          - batch_size:         DataLoader batch size
          - num_workers:        DataLoader workers
          - limit:              Max subjects (for quick tests)
          - normalize:          Z-score normalize features
    """

    def __init__(self, args):
        self.args = args
        # All cohorts (FCD Bonn, IDEAS, IXI) live under one FastSurfer subjects
        # directory; each subject folder is prefixed with its cohort name.
        self.subjects_dir = getattr(args, "subjects_dir_ext",
                                    getattr(args, "subjects_dir",
                                            "data/fastsurfer_subjects"))
        self.hemispheres = getattr(args, "hemispheres", list(HEMIS))
        self.batch_size = getattr(args, "batch_size", 1)
        self.num_workers = getattr(args, "num_workers", 0)
        self.val_samples_in_train = getattr(args, "val_samples_in_train", 5000)

        eval_dataset = getattr(args, "eval_dataset", "fcdbonn")
        train_dataset = getattr(args, "train_dataset", None)

        # ── Load EVAL data (external cohort: fcdbonn or idea) ──
        print(f"Loading evaluation dataset: {eval_dataset} "
              f"(from {self.subjects_dir})...")
        t0 = time.time()
        self.lst_dicts = self._load_ext_dataset(eval_dataset)
        print(f"  Loaded {len(self.lst_dicts)} subjects in {time.time()-t0:.1f}s")

        # ── Load TRAIN data (only if different from the eval cohort) ──
        # Detection/classification (mode detect/classify) never need training
        # data; only train/baseline/visualize_results do.
        self.train_dicts_ext = None
        self._combined_mode = False
        mode = getattr(args, "mode", "train")
        if train_dataset and train_dataset != eval_dataset \
                and mode in ("train", "baseline", "visualize_results"):
            if train_dataset == "ixi":
                print(f"\nLoading IXI healthy training data...")
                self.train_dicts_ext = self._load_ixi_training_data()
                print(f"  Loaded {len(self.train_dicts_ext)} IXI healthy subjects")
            elif train_dataset == "fcdbonn":
                print(f"\nLoading FCD Bonn healthy training data...")
                fcd = self._load_ext_dataset("fcdbonn")
                self.train_dicts_ext = [d for d in fcd
                                        if len(d.get("seg_masks", [])) == 0]
                print(f"  Loaded {len(self.train_dicts_ext)} FCD Bonn healthy subjects")
            elif train_dataset == "combined":
                # Combined = pooled healthy controls from FCD Bonn + IXI.
                print(f"\nLoading combined healthy training data (FCD Bonn + IXI)...")
                fcd = self._load_ext_dataset("fcdbonn")
                self._combined_fcd = [d for d in fcd
                                      if len(d.get("seg_masks", [])) == 0]
                self._combined_ixi = self._load_ixi_training_data()
                print(f"  FCD Bonn: {len(self._combined_fcd)} healthy")
                print(f"  IXI:      {len(self._combined_ixi)} healthy")
                self.train_dicts_ext = []     # non-None sentinel
                self._combined_mode = True

        # ── Optional limit ──
        limit = getattr(args, "limit", None)
        if limit:
            self.lst_dicts = self.lst_dicts[:limit]

        # ── Split eval cohort (no training subjects: train=[]) ──
        self.train_dicts, self.val_dicts, self.test_dicts = self._split_data()
        self.eval_train_dicts = list(self.train_dicts)

        # ── Override train split with the chosen training cohort ──
        if self._combined_mode:
            f_train, f_val, f_test = self._split_external_train(self._combined_fcd)
            i_train, i_val, i_test = self._split_external_train(self._combined_ixi)
            self.train_dicts = f_train + i_train
            self.train_val_dicts = f_val + i_val
            self.train_test_dicts = f_test + i_test
            print(f"\n  Combined train: {len(self.train_dicts)} "
                  f"(fcdbonn={len(f_train)}, ixi={len(i_train)})")
            print(f"  Combined val:   {len(self.train_val_dicts)}")
            print(f"  Combined test:  {len(self.train_test_dicts)}")
        elif self.train_dicts_ext:
            (self.train_dicts,
             self.train_val_dicts,
             self.train_test_dicts) = self._split_external_train(self.train_dicts_ext)


        # ── Preload shared mesh ──
        print("Loading shared fsaverage mesh...")
        self.mesh_cache = {}
        for hemi in self.hemispheres:
            edge_index, positions = load_fsaverage_mesh(self.subjects_dir, hemi)
            edge_attr = compute_edge_features(edge_index, positions)
            self.mesh_cache[hemi] = {
                "edge_index": edge_index,
                "positions": positions,
                "edge_attr": edge_attr,
            }

        # ── Build datasets ──
        print(f"\nBuilding datasets (morpho only)...")

        # Normalization stats live in the run's --val_output_dir, alongside the
        # healthy baseline and the detection/classification results, so a results
        # folder is self-contained. The filename is keyed by the training cohort.
        # A legacy location (validation_results_cyclevae/) is still accepted on
        # load for backward compatibility with older runs.
        train_ds = train_dataset or eval_dataset
        out_dir = pathlib.Path(getattr(args, "val_output_dir",
                                       "validation_results_cyclevae"))
        norm_stats_path = out_dir / f"norm_stats_{train_ds}.pt"
        legacy_norm_stats_path = (pathlib.Path("validation_results_cyclevae")
                                  / f"norm_stats_{train_ds}.pt")

        def _load_norm_stats():
            """Load norm stats from val_output_dir, else the legacy location."""
            import torch
            for p in (norm_stats_path, legacy_norm_stats_path):
                if p.exists():
                    print(f"  Loaded norm stats from {p}")
                    return torch.load(str(p), weights_only=False)
            raise FileNotFoundError(
                f"\nNo norm stats found at {norm_stats_path}\n"
                f"(also checked legacy {legacy_norm_stats_path}).\n"
                f"Run --mode train or --mode baseline first.\n")

        norm_stats = None

        if mode in ("train", "baseline", "visualize_results") and len(self.train_dicts) > 0:
            # All training cohorts (fcdbonn / ixi / combined) live under the
            # same external FastSurfer subjects directory.
            train_sd = self.subjects_dir

            self.train_dataset = self._build_dataset(
                self.train_dicts, subjects_dir_override=train_sd)

            if mode in ("train", "baseline"):
                norm_stats = self.train_dataset.get_norm_stats()
                if norm_stats.get("mean") is not None:
                    import torch
                    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(norm_stats, str(norm_stats_path))
                    print(f"  Saved norm stats to {norm_stats_path}")
            else:
                # visualize_results: reuse saved norm stats (never overwrite).
                norm_stats = _load_norm_stats()
        else:
            self.train_dataset = self._build_dataset([])
            norm_stats = _load_norm_stats()

        self.val_dataset = self._build_dataset(self.val_dicts, norm_stats=norm_stats)
        self.test_dataset = self._build_dataset(self.test_dicts, norm_stats=norm_stats)

        self._print_summary()

    # ─────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────

    def _load_ext_dataset(self, dataset_name: str) -> List[dict]:
        """Load external dataset (FCDBONN, IDEA, etc.) from fastsurfer_subjects dir.

        Scans subjects_dir for dirs matching {prefix}* pattern.
        Finds seg masks from original data dir for ground truth.

        For IDEAS: optionally filters to FCD-only cases via
        --ideas_filter {all, fcd, fcd_type2, fcd_including_dual}

        Returns a list of subject dicts.
        """
        # Map eval_dataset name → directory prefix.
        # (IDEAS subject dirs are prefixed ideas__sub-*, not idea__)
        PREFIX_MAP = {
            "idea": "ideas__",
            "fcdbonn": "fcdbonn__",
        }
        prefix = PREFIX_MAP.get(dataset_name.lower(), dataset_name.lower() + "__")
        subjects_dir = pathlib.Path(self.subjects_dir)
        ext_data_dir = getattr(self.args, "ext_data_dir",
                               str(pathlib.Path(__file__).parent.parent / "data" / dataset_name.upper()))

        # Load IDEAS pathology filter (FCD-only, etc.)
        allowed_ids = None
        if dataset_name.lower() == "idea":
            allowed_ids = self._load_ideas_pathology_filter(ext_data_dir)

        lst_dicts = []
        for sd in sorted(subjects_dir.iterdir()):
            if not sd.is_dir() or not sd.name.startswith(prefix):
                continue
            sid = sd.name

            # Check morpho.npz exists for both hemis
            if not (sd / "lh.morpho.npz").exists() or not (sd / "rh.morpho.npz").exists():
                continue

            # Find seg mask from original data dir (dataset-specific paths)
            sub_name = sid[len(prefix):]  # e.g. "sub-00151" or "sub-1"

            # IDEAS FCD filter: skip if not in allowed pathology list
            if allowed_ids is not None:
                try:
                    sub_num = int(sub_name.replace("sub-", ""))
                except ValueError:
                    continue
                if sub_num not in allowed_ids:
                    continue

            seg_masks = self._find_seg_masks(dataset_name, ext_data_dir, sub_name)

            has_anomaly = len(seg_masks) > 0

            d = {
                "study_uid": sid,
                "study_befund": "mit Befund" if has_anomaly else "ohne Befund",
                "city": None,
                "seg_masks": seg_masks,
                "keyword": dataset_name.lower(),
            }
            lst_dicts.append(d)

        n_patho = sum(1 for d in lst_dicts if d["seg_masks"])
        n_healthy = len(lst_dicts) - n_patho
        filter_tag = ""
        if allowed_ids is not None:
            filter_tag = f" [filter: {len(allowed_ids)} allowed IDs]"
        print(f"  {dataset_name}: {len(lst_dicts)} subjects "
              f"({n_healthy} healthy, {n_patho} with seg mask){filter_tag}")

        return lst_dicts

    def _load_ixi_training_data(self) -> List[dict]:
        """Load IXI subjects as healthy training data."""
        subjects_dir_ext = getattr(self.args, "subjects_dir_ext",
                                   "data/fastsurfer_subjects")
        sd = pathlib.Path(subjects_dir_ext)

        ixi_dicts = []
        for d in sorted(sd.iterdir()):
            if not d.is_dir() or not d.name.startswith("ixi__"):
                continue
            # Check morpho.npz exists
            if not (d / "lh.morpho.npz").exists():
                continue
            parts = d.name.replace("ixi__", "").split("-")
            scanner = parts[1] if len(parts) >= 2 else "UNKNOWN"
            ixi_dicts.append({
                "study_uid": d.name,
                "study_befund": "ohne Befund",
                "city": f"IXI_{scanner}",  # used for stratification
                "seg_masks": [],
                "keyword": "ixi",
            })
        return ixi_dicts

    def _load_ideas_pathology_filter(self, ext_data_dir: str) -> Optional[Set[int]]:
        """Load IDEAS pathology filter from Metadata_Release_Anon.csv.

        Controlled by --ideas_filter arg:
          'all'                 → None (accept everything, default)
          'fcd'                 → pure FCD pathology only (40 subjects)
          'fcd_type2'           → FCD Type IIa + IIb only (38 subjects)
          'fcd_including_dual'  → FCD + DUAL cases mentioning FCD (42 subjects)
          'dnt'                 → DNT (low-grade cortical tumors, 52 subjects)
          'cav'                 → Cavernomas (33 subjects)
          'gliosis'             → Gliosis cases from OTHER (25+ subjects)

        Returns:
            Set of allowed integer subject IDs, or None to accept all.
        """
        filter_mode = getattr(self.args, "ideas_filter", "all").lower()
        if filter_mode == "all":
            return None

        # Find metadata CSV
        csv_candidates = [
            pathlib.Path(ext_data_dir) / "Metadata_Release_Anon.csv",
            pathlib.Path(ext_data_dir).parent / "IDEAS" / "Metadata_Release_Anon.csv",
            pathlib.Path(ext_data_dir) / "IDEAS" / "Metadata_Release_Anon.csv",
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            print(f"  WARNING: --ideas_filter={filter_mode} but Metadata_Release_Anon.csv not found")
            print(f"  Searched: {[str(p) for p in csv_candidates]}")
            return None

        try:
            import pandas as pd
        except ImportError:
            print(f"  WARNING: --ideas_filter requires pandas; accepting all subjects")
            return None

        df = pd.read_csv(csv_path)
        allowed = set()

        if filter_mode == "fcd":
            allowed.update(df[df["Pathology"] == "FCD"]["ID"].astype(int).tolist())

        elif filter_mode == "fcd_type2":
            fcd = df[df["Pathology"] == "FCD"].copy()
            memo = fcd["OP MEMO"].astype(str).str.lower()
            is_type2 = memo.str.contains("type ii") & ~memo.str.contains("type iii")
            allowed.update(fcd[is_type2]["ID"].astype(int).tolist())

        elif filter_mode == "fcd_including_dual":
            allowed.update(df[df["Pathology"] == "FCD"]["ID"].astype(int).tolist())
            dual = df[df["Pathology"] == "DUAL"].copy()
            dual_fcd = dual[dual["OP MEMO"].astype(str).str.contains("FCD", case=False, na=False)]
            allowed.update(dual_fcd["ID"].astype(int).tolist())

        elif filter_mode == "dnt":
            allowed.update(df[df["Pathology"] == "DNT"]["ID"].astype(int).tolist())

        elif filter_mode == "cav":
            allowed.update(df[df["Pathology"] == "CAV"]["ID"].astype(int).tolist())

        elif filter_mode == "gliosis":
            other = df[df["Pathology"] == "OTHER"].copy()
            memo = other["OP MEMO"].astype(str).str.lower()
            is_gliosis = memo.str.contains("gliosis")
            allowed.update(other[is_gliosis]["ID"].astype(int).tolist())

        elif filter_mode == "mcd":
            # MCD / heterotopia from OTHER — cortical malformations (FCD-adjacent)
            other = df[df["Pathology"] == "OTHER"].copy()
            memo = other["OP MEMO"].astype(str).str.lower()
            is_mcd = memo.str.contains("mcd|heterotopia|polymicrogyria|hamartoma")
            allowed.update(other[is_mcd]["ID"].astype(int).tolist())

        elif filter_mode == "hs":
            allowed.update(df[df["Pathology"] == "HS"]["ID"].astype(int).tolist())

        elif filter_mode == "gl":
            allowed.update(df[df["Pathology"] == "GL"]["ID"].astype(int).tolist())

        elif filter_mode == "all_cortical":
            # FCD + DNT + CAV + GL + MCD — all pathologies expected on cortical surface
            allowed.update(df[df["Pathology"] == "FCD"]["ID"].astype(int).tolist())
            allowed.update(df[df["Pathology"] == "DNT"]["ID"].astype(int).tolist())
            allowed.update(df[df["Pathology"] == "CAV"]["ID"].astype(int).tolist())
            allowed.update(df[df["Pathology"] == "GL"]["ID"].astype(int).tolist())
            other = df[df["Pathology"] == "OTHER"].copy()
            memo = other["OP MEMO"].astype(str).str.lower()
            is_mcd = memo.str.contains("mcd|heterotopia|polymicrogyria|hamartoma")
            allowed.update(other[is_mcd]["ID"].astype(int).tolist())

        else:
            print(f"  WARNING: unknown --ideas_filter={filter_mode}, accepting all")
            return None

        print(f"  IDEAS filter '{filter_mode}': {len(allowed)} subject IDs from metadata")
        return allowed

    def _find_seg_masks(self, dataset_name: str, ext_data_dir: str,
                        sub_name: str) -> List[str]:
        """Find segmentation masks for a subject (dataset-specific paths).

        Args:
            dataset_name: "fcdbonn", "idea", etc.
            ext_data_dir: Root path to the raw dataset
            sub_name: Subject name without prefix (e.g. "sub-00151", "sub-1")
        """
        ds = dataset_name.lower()

        if ds == "fcdbonn":
            # FCDBONN: ../data/FCDBONN/sub-XXXXX/anat/*_roi.nii.gz
            anat_dir = pathlib.Path(ext_data_dir) / sub_name / "anat"
            if anat_dir.exists():
                return sorted(glob.glob(str(anat_dir / "*_roi.nii.gz")))

        elif ds == "idea":
            # IDEA: ../data/IDEAS/ds005602_masks/<num>/<num>_MaskInOrig.nii.gz
            # sub_name = "sub-1" → sub_num = "1"
            sub_num = sub_name.replace("sub-", "")
            mask_base = pathlib.Path(ext_data_dir).parent / "IDEAS" / "ds005602_masks"
            # Also try ext_data_dir directly if user sets --ext_data_dir to the masks dir
            for base in [mask_base, pathlib.Path(ext_data_dir) / "ds005602_masks"]:
                mask_path = base / sub_num / f"{sub_num}_MaskInOrig.nii.gz"
                if mask_path.exists():
                    return [str(mask_path)]

        return []

    # ─────────────────────────────────────────────────────
    # Splitting logic (mirrors old MRIDataModule)
    # ─────────────────────────────────────────────────────

    def _split_data(self) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Split the evaluation cohort for anomaly detection.

          - Train: empty (the model is trained on a separate healthy cohort;
                   see --train_dataset). The eval cohort contributes no training
                   subjects, so the unsupervised detector never sees lesions.
          - Validation + Test: a deterministic 50/50 split of the cohort,
                   stratified by healthy/pathological. random_state=42.

        NOTE ON NAMING: the two halves are assigned so that ``--split val``
        and ``--split test`` carry their intended cohorts. The partitioning
        itself is unchanged (same seed) — only the names attached to each half
        are fixed: the first half is the **test** set, the second half is the
        **validation** set.
        """
        all_subjects = self.lst_dicts
        if len(all_subjects) >= 2:
            # Stratify so both halves have proportional healthy/pathological.
            strat_labels = [1 if len(d.get("seg_masks", [])) > 0 else 0
                            for d in all_subjects]
            from collections import Counter
            label_counts = Counter(strat_labels)
            can_stratify = all(c >= 2 for c in label_counts.values())
            try:
                first_half, second_half = sk_model_selection.train_test_split(
                    all_subjects, test_size=0.5, random_state=42,
                    stratify=strat_labels if can_stratify else None)
            except ValueError:
                first_half, second_half = sk_model_selection.train_test_split(
                    all_subjects, test_size=0.5, random_state=42)
        else:
            first_half, second_half = all_subjects, []

        # Naming fix (seeds unchanged): first half = test, second half = validation.
        all_test = first_half
        all_val = second_half

        print(f"\n  Split (eval only, no train): "
              f"val={len(all_val)}, test={len(all_test)}")
        for name, lst in [("val", all_val), ("test", all_test)]:
            n_p = sum(1 for d in lst if len(d.get("seg_masks", [])) > 0)
            n_h = len(lst) - n_p
            print(f"    {name}: {n_h} healthy, {n_p} pathological")

        return [], all_val, all_test

    def _split_external_train(self, train_dicts: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
        """Split a healthy training cohort 80/10/10 into train/val/test.

        random_state is fixed (42) and unchanged. As in ``_split_data``, the
        two 10% held-out halves are named so the first is **test** and the
        second is **validation**.
        """
        cities = [d.get("city", "OTHER") for d in train_dicts]
        from collections import Counter
        city_counts = Counter(cities)
        can_stratify = all(c >= 2 for c in city_counts.values()) and len(city_counts) > 1

        try:
            train, temp = sk_model_selection.train_test_split(
                train_dicts, test_size=0.2, random_state=42,
                stratify=cities if can_stratify else None)
            temp_cities = [d.get("city", "OTHER") for d in temp]
            first_half, second_half = sk_model_selection.train_test_split(
                temp, test_size=0.5, random_state=42,
                stratify=temp_cities if can_stratify else None)
        except ValueError:
            train, temp = sk_model_selection.train_test_split(
                train_dicts, test_size=0.2, random_state=42)
            first_half, second_half = sk_model_selection.train_test_split(
                temp, test_size=0.5, random_state=42)

        # Naming fix (seeds unchanged): first half = test, second half = validation.
        test, val = first_half, second_half

        print(f"\n  External train split: "
              f"train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    # ─────────────────────────────────────────────────────
    # Dataset construction
    # ─────────────────────────────────────────────────────

    def _build_dataset(self, data_dicts: List[dict],
                       norm_stats: Optional[dict] = None,
                       subjects_dir_override: str = None) -> CorticalSurfaceDataset:
        sd = subjects_dir_override or self.subjects_dir
        return CorticalSurfaceDataset(
            data_dicts=data_dicts,
            subjects_dir=sd,
            hemispheres=self.hemispheres,
            patch_size=getattr(self.args, "patch_size", 5000),
            mask_ratio_range=getattr(self.args, "mask_ratio_range", (0.15, 0.40)),
            normalize=getattr(self.args, "normalize", True),
            mesh_cache=self.mesh_cache,
            norm_stats=norm_stats,
            atlas=getattr(self.args, "atlas", "dkt"),
        )

    # ─────────────────────────────────────────────────────
    # DataLoaders
    # ─────────────────────────────────────────────────────

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return train, val, test DataLoaders."""
        if len(self.train_dataset) > 0:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
                num_workers=self.num_workers,
                sampler=RandomSampler(
                    self.train_dataset,
                    replacement=True,
                    num_samples=getattr(self.args, "samples_per_epoch", 100000),
                ),
            )
        else:
            train_loader = None
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=RandomSampler(
                self.val_dataset,
                replacement=True,
                num_samples=self.val_samples_in_train,
            ),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) if len(self.test_dataset) > 0 else None

        return train_loader, val_loader, test_loader

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────

    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"Surface Data Module Summary")
        print(f"{'='*60}")
        print(f"  Train graphs:       {len(self.train_dataset)}")
        print(f"  Val graphs:         {len(self.val_dataset)}")
        print(f"  Test graphs:        {len(self.test_dataset)}")
        # Get info from first available dataset
        ref_ds = self.train_dataset if len(self.train_dataset) > 0 else (
            self.val_dataset if len(self.val_dataset) > 0 else self.test_dataset)
        print(f"  Features/vertex:    {ref_ds.num_node_features}")
        print(f"  Feature names:      {ref_ds.feature_names}")
        print(f"  Vertices/graph:     {N_FSAVERAGE_VERTICES:,}")
        patch_size = getattr(self.args, "patch_size", 5000)
        if patch_size > 0 and patch_size < N_FSAVERAGE_VERTICES:
            print(f"  Patch size:         {patch_size:,} nodes (from {N_FSAVERAGE_VERTICES:,})")
        else:
            print(f"  Patch size:         full hemisphere ({N_FSAVERAGE_VERTICES:,})")
        print(f"  Features:           morpho only")
        print(f"  Hemispheres:        {self.hemispheres}")

        if len(ref_ds) > 0:
            sample = ref_ds[0]
            print(f"\n  Sample graph:")
            print(f"    x:           {sample.x.shape}")
            print(f"    edge_index:  {sample.edge_index.shape}")
            print(f"    edge_attr:   {sample.edge_attr.shape}")
            print(f"    pos:         {sample.pos.shape}")
            print(f"    atlas_label: unique={sample.atlas_label.unique().shape[0]}")