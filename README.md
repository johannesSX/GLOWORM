# GLOWORM — Graph Cycle-VAE for Epileptogenic Lesion Detection

Unsupervised detection of epileptogenic cortical lesions (FCD, DNT, cavernoma, …) from **T1-only** MRI, on FreeSurfer/FastSurfer cortical surfaces.

The model is a **cycle-consistent dual graph-VAE** trained on **healthy controls only**. It learns the normal mapping between cortical *geometry* and *structure* and flags vertices where that mapping breaks down:

```
Path A : geometry (curv, sulc) ──▶ structure (thickness, area, wg_pct)
Path B : structure             ──▶ geometry
Cycle  : geometry ─▶ structure_pred ─▶ geometry_recon   (and the reverse)

anomaly(v) = cross-modal reconstruction error(v) + cycle inconsistency(v)
```

Per-vertex anomaly scores are z-scored against a healthy per-region baseline, clustered on the mesh graph, and matched against a ground-truth segmentation mask in MNI space.

> Morphometric (T1-derived) features only — no FLAIR / T2 / SWI.

> _This repository is a public, readability-focused refactor of the original research code, prepared with help from Claude Opus 4.8. The underlying methods, models, and results are unchanged._

---

## 1. Repository layout

```
cycle_vae.py            Graph Cycle-VAE model (encoder/decoder/VAE paths)
gatedgcn_layer.py       Residual GatedGCN message-passing layer
laplace_pos_encoder.py  Laplacian positional encoding
mni_pos_encoder.py      MNI-coordinate + atlas positional encoding
dataset.py              PyG dataset: fsaverage mesh + BFS patches + features
build_dataset.py        SurfaceDataModule: cohort loading + train/val/test split
pl.py                   Training loop
validation.py           baseline / detection / classification
visualize_results.py    glass-brain figures
run_gloworm.py          command-line entry point
requirements.txt
```

Model **checkpoints from the paper are distributed separately** — each comes with its normalization stats and healthy baseline. See [**Section 6 — Using the paper checkpoints**](#6-using-the-paper-checkpoints).

---

## 2. Preprocessing (separate repository)

This repo expects **already-preprocessed** surface features. The pipeline that turns a raw T1 MRI into the per-subject `*.morpho.npz` files (FastSurfer reconstruction → resampling to `fsaverage` → morphometric feature stacking) lives in its own repository:

> **Preprocessing:** `https://github.com/johannesSX/SurfPrep`

Run that pipeline first for every subject, then arrange the outputs as below.

---

## 3. Data layout

Everything lives under **one** FastSurfer subjects directory (`--subjects_dir_ext`, default `data/fastsurfer_subjects`). Each cohort's subjects are prefixed with the cohort name:

```
data/fastsurfer_subjects/
├── fsaverage_common/                 # shared mesh (produced by preprocessing)
│   ├── lh.positions.npy              #   (163842, 3) vertex coordinates
│   ├── rh.positions.npy
│   ├── lh.edge_index.pt              #   cached mesh connectivity (optional)
│   └── rh.edge_index.pt
├── fsaverage/                        # standard FreeSurfer fsaverage
│   └── surf/{lh,rh}.white, {lh,rh}.sphere.reg, ...
│
├── fcdbonn__sub-00055/               # one folder per subject, prefix = cohort
│   ├── lh.morpho.npz                 #   (163842, 5) thickness,curv,sulc,area,wg_pct
│   ├── rh.morpho.npz
│   ├── fsaverage_features/           #   per-subject DKT atlas on fsaverage
│   │   ├── lh.aparc.DKTatlas.fsaverage.annot
│   │   └── rh.aparc.DKTatlas.fsaverage.annot
│   └── mri/                          #   needed for seg-mask → MNI + volume output
│       ├── orig.mgz   (or rawavg.mgz)
│       └── transforms/talairach.xfm
│
├── ideas__sub-1/                     # IDEAS cohort (same structure)
│   └── ...
└── ixi__IXI002-Guys-0828/            # IXI healthy controls (no seg mask)
    └── ...
```

`lh.morpho.npz` / `rh.morpho.npz` each store a `data` array of shape `(163842, num_morpho)` plus optional `feature_names`. The default feature order is `thickness, curv, sulc, area, wg_pct` (`--num_morpho 5`).

### Ground-truth segmentation masks

Lesion masks (used only for evaluation, never for training) stay in the **raw** dataset directory pointed to by `--ext_data_dir`:

```
# FCD Bonn
data/FCDBONN/sub-00055/anat/*_roi.nii.gz

# IDEAS
data/IDEAS/ds005602_masks/<num>/<num>_MaskInOrig.nii.gz
data/IDEAS/Metadata_Release_Anon.csv      # for --ideas_filter
```

Masks are transformed voxel → scanner-RAS (mask affine) → MNI305 (`talairach.xfm`) and matched against detected clusters in MNI space, so no surface projection of the mask is required.

---

## 4. Validation / test naming

`--split val` selects the **validation** set and `--split test` selects the **test** set. The eval cohort is split 50/50 with a fixed seed (`random_state=42`); training cohorts are split 80/10/10 with the same seed. The split seeds are fixed for reproducibility.

Tune any classification threshold on `val` and report on `test` (`--threshold_split val` while running `--split test`).

---

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch / PyTorch-Geometric wheels are hardware-specific — install the build that matches your CUDA version from the official indices first if needed.

---

## 6. Using the paper checkpoints

Pretrained checkpoints from the paper are accessible through [(soon be available)](). For each model **three** files travel together and must all be placed before you can evaluate:

| Provided file | Put it at | Why |
|---|---|---|
| `best.pt` | anywhere — pass via `--val_checkpoint` | the model weights |
| `norm_stats_<train_dataset>.pt` | `<val_output_dir>/norm_stats_<train_dataset>.pt` | **required** per-feature normalization; every eval mode reloads it (keyed by `--train_dataset`) and errors without it |
| `healthy_baseline.json` | `<val_output_dir>/healthy_baseline.json` | per-region healthy reference used to z-score the anomaly scores |

`<train_dataset>` is the cohort the model was trained on (e.g. `ixi` → the file is `norm_stats_ixi.pt`). If the model also shipped a `healthy_baseline_vertex.npz`, drop it next to `healthy_baseline.json`.

With all three in place you can **skip both training and the baseline step** and go straight to detection / classification:

```bash
# place the provided files where the code expects them
mkdir -p results/fcdbonn
cp norm_stats_ixi.pt      results/fcdbonn/
cp healthy_baseline.json  results/fcdbonn/

# detect — no --mode baseline needed
python run_gloworm.py --mode detect \
    --train_dataset ixi --eval_dataset fcdbonn --split val \
    --subjects_dir_ext /path/to/fastsurfer_subjects_ext \
    --ext_data_dir /path/to/FCDBONN \
    --val_checkpoint best.pt \
    --val_output_dir results/fcdbonn \
    --max_clusters 3 --cluster_match_distance 20
```

**Recompute the baseline instead** when you change anything it depends on. A provided `healthy_baseline.json` is valid only for the exact checkpoint + `--atlas` + eval cohort it was generated with. If you switch atlas, switch eval dataset, or use your own retrained checkpoint, delete it and run:

```bash
python run_gloworm.py --mode baseline \
    --train_dataset ixi --eval_dataset fcdbonn \
    --subjects_dir_ext /path/to/fastsurfer_subjects_ext \
    --ext_data_dir /path/to/FCDBONN \
    --val_checkpoint best.pt --val_output_dir results/fcdbonn
```

> Note: the BFS patch sampling at score time is not seeded, so detection z-scores (and the resulting metrics) vary slightly from run to run. Using the **provided** `healthy_baseline.json` removes the baseline's share of that variance and reproduces the paper numbers most closely.

---

## 7. How to run

All commands share `--subjects_dir_ext`, `--ext_data_dir`, `--eval_dataset`, and `--val_output_dir`. Pick a results directory per eval cohort and reuse it across baseline → detect → classify → visualize.

> **Important — keep one results folder per experiment.** Pass the *same* `--val_output_dir` **and** the *same* `--train_dataset` on every mode (train included). Everything a run needs then lives in that one folder: `norm_stats_<train_dataset>.pt`, `healthy_baseline.json`, `detection/`, `classification/`, `viz_results/`. The normalization stats are keyed by `--train_dataset` (e.g. `norm_stats_ixi.pt`); omitting `--train_dataset` on a later mode makes it look for `norm_stats_<eval_dataset>.pt` and fail. (For backward compatibility, a stats file in the legacy `validation_results_cyclevae/` directory is still accepted on load.)

### 7.1 Train (on healthy controls)

```bash
python run_gloworm.py --mode train \
    --train_dataset ixi \
    --eval_dataset fcdbonn \
    --subjects_dir_ext data/fastsurfer_subjects \
    --val_output_dir results/fcdbonn \
    --epochs 50 --batch_size 16 --patch_size 5000 --lr 2e-4
```

`--train_dataset` options: `ixi`, `fcdbonn` (its healthy controls), or `combined` (FCD Bonn + IXI healthy pooled). Checkpoints are written to `checkpoints_cyclevae/{best,final,epoch_N}.pt`; the best checkpoint is selected on the validation loss.

### 7.2 Healthy baseline (per-region reference statistics)

Required before detection/classification — it computes the healthy mean/SD per atlas region that anomaly scores are z-scored against.

```bash
python run_gloworm.py --mode baseline \
    --train_dataset ixi --eval_dataset fcdbonn \
    --subjects_dir_ext data/fastsurfer_subjects \
    --val_checkpoint checkpoints_cyclevae/best.pt \
    --val_output_dir results/fcdbonn
```

Writes `results/fcdbonn/healthy_baseline.json`.

### 7.3 Detection (cluster-level, MELD-style)

```bash
python run_gloworm.py --mode detect \
    --train_dataset ixi --eval_dataset fcdbonn --split val \
    --subjects_dir_ext data/fastsurfer_subjects \
    --ext_data_dir data/FCDBONN \
    --val_checkpoint checkpoints_cyclevae/best.pt \
    --val_output_dir results/fcdbonn \
    --max_clusters 3 --cluster_match_distance 20
```

Writes `results/fcdbonn/detection/detection_results.json` and per-subject z-maps / NIfTI / HTML overlays. For IDEAS add `--eval_dataset idea` and an optional `--ideas_filter fcd_type2` (etc.).

### 7.4 Classification (subject-level, AUROC)

```bash
# establish the threshold on validation …
python run_gloworm.py --mode classify \
    --train_dataset ixi --eval_dataset fcdbonn --split val --threshold_split self \
    --val_checkpoint checkpoints_cyclevae/best.pt \
    --val_output_dir results/fcdbonn

# … then report on test using the validation threshold
python run_gloworm.py --mode classify \
    --train_dataset ixi --eval_dataset fcdbonn --split test --threshold_split val \
    --val_checkpoint checkpoints_cyclevae/best.pt \
    --val_output_dir results/fcdbonn
```

Writes `results/fcdbonn/classification/<cohort>/classification_results.json`.

### 7.5 Glass-brain visualization

Reads the `detection_results.json` produced in 7.3.

```bash
python run_gloworm.py --mode visualize_results --vis_result_mode glassbrain \
    --train_dataset ixi --eval_dataset fcdbonn --split test \
    --ext_data_dir data/FCDBONN \
    --subjects_dir_ext data/fastsurfer_subjects \
    --val_output_dir results/fcdbonn
```

One PNG per subject under `results/fcdbonn/viz_results/glassbrain/`: the GT mask region (blue outline), the top clusters (green = TP, magenta = FP) over the projected anomaly z-map. Omit `--vis_subject` to render every subject in the detection results.

---

## 8. Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--mode` | `train` | `train`, `baseline`, `detect`, `classify`, `visualize_results` |
| `--train_dataset` | — | `ixi`, `fcdbonn`, `combined` |
| `--eval_dataset` | `fcdbonn` | `fcdbonn`, `idea` |
| `--split` | `val` | `val` or `test` |
| `--threshold_split` | `self` | `self`, `val`, `test` (classification) |
| `--ideas_filter` | `all` | IDEAS pathology subset (`fcd`, `fcd_type2`, `dnt`, `cav`, …) |
| `--atlas` | `dkt` | `dkt`, `destrieux`, `hcp` |
| `--patch_size` | `5000` | BFS patch size (0 = full hemisphere) |
| `--cluster_z_threshold` | `2.5` | z-threshold for cluster formation |
| `--max_clusters` | `3` | clusters kept per subject |
| `--cluster_match_distance` | `20` | TP distance to GT in mm |
| `--exclude_regions` | `[]` | atlas region IDs to drop (artifact regions) |
| `--val_checkpoint` | `checkpoints_cyclevae/best.pt` | model weights |
| `--val_output_dir` | `validation_results_cyclevae/` | results directory |

Run `python run_gloworm.py --help` for the full list.

---

## 9. Reproducibility notes

- Use **one `--val_output_dir` per experiment**, passed to every mode (train → baseline → detect → classify → visualize). It then holds the normalization stats, the healthy baseline, and all results together.
- Splits use a fixed seed (`random_state=42`).