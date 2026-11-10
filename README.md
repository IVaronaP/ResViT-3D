# ResViT-3D — volumetric brightfield → fibrillarin fluorescence prediction

A 3D adaptation of [ResViT](https://github.com/icon-lab/ResViT) for label-free prediction
of **nucleolar fibrillarin** from **3D brightfield** microscopy volumes.

```
3D brightfield volume  ──►  ResViT-3D  ──►  predicted 3D fibrillarin volume
```

The task is **image-to-image regression** (continuous fluorescence intensity), not
segmentation. No binary masks are used or produced anywhere in this pipeline.

Data: Allen Institute for Cell Science hiPSC collection, **FBL-mEGFP** line
(fibrillarin tagged with mEGFP). Fibrillarin marks the **nucleolus** — the dense
fibrillar component — not the whole nucleus.

---

## 1. Why a 3D version

The original ResViT is 2D. A 2D slice cuts an arbitrary section through a nucleolus,
which is a compact 3D body of roughly 1–3 µm. At the 0.29 µm axial voxel size of this
dataset a single nucleolus spans ~4–10 slices, so the volumetric context is genuinely
informative rather than decorative. This repository keeps the full Z extent of every
acquisition and lets the network convolve across depth.

## 2. A data bug found in the 2D pipeline (important)

While porting, the channel selection of the previous 2D pipeline was found to be wrong.
The CZI files have 7 channels:

| C | Name | Structure |
|---|---|---|
| 0, 2, 4 | `CMDRP_*`, `EGFP_*`, `H3342_*` | near-empty short-exposure duplicates |
| 1 | `CMDRP` | CellMask Deep Red — plasma membrane |
| **3** | **`EGFP`** | **fibrillarin — nucleolus (the correct target)** |
| 5 | `H3342` | Hoechst 33342 — DNA / whole nucleus |
| **6** | **`Bright_100`** | **brightfield (the input)** |

The 2D exporter selected the target as `C - 2`, i.e. **channel 5 (`H3342`, DNA)** instead
of channel 3 (`EGFP`, fibrillarin). Measured on
`3500001432_100X_20171020_1-Scene-06-P7-E04.czi`, whole-volume p99.5/p50 contrast:

```
C1 CMDRP     2.72      C3 EGFP      3.33   <- correct target
C5 H3342     1.18      C6 Bright    1.19   <- input (low ratio is expected: brightfield
C0/C2/C4  1.05-1.23           information is in texture, not sparse bright spots)
```

`H3342` is essentially flat in these acquisitions (1.18), and its narrow dynamic range
explains an independent observation: the 8-bit PNGs of the old 2D datasets used only
85–129 of 256 grey levels.

**Consequence:** every dataset and checkpoint derived from the old exporter was trained
against the DNA channel, i.e. a different biological structure. This repository selects
channels **by name** (`EGFP` exactly, and any channel starting with `Bright`), which is
also robust to the `_1`/`_2` suffix variation across the 80 files.

## 3. What had to change for 3D

The transformer needed **no changes at all**: `Attention`, `Mlp`, `Block` and `Encoder`
operate on `(B, N, hidden)` token sequences and are byte-identical to the 2D original.
Only the convolutional shell and the token↔volume reshape differ.

| Component | Change |
|---|---|
| encoder / decoder / residual blocks / ART blocks / discriminator | `Conv2d`→`Conv3d`, `ConvTranspose2d`→`ConvTranspose3d`, `BatchNorm2d`→`BatchNorm3d` |
| reflection padding | → `ReplicationPad3d`. **`nn.ReflectionPad3d` does not exist in torch 1.7.1** (added in 1.10), so this substitution is forced, not stylistic |
| patch embedding | explicit `patch_size=(1,1,1)` and a per-axis grid `(Z/16, Y/16, X/16)`, replacing the 2D-only `img_size//16//grid` arithmetic |
| token → volume | `d, h, w = embeddings.grid` instead of `int(np.sqrt(n_patch))`, which assumed a *square* grid |
| positional embeddings | the pretrained 14×14 grid is resized to `(gh, gw)` and replicated across the `gd` depth positions. Replication sets only the **initial** value: the table holds one independent vector per `(z,y,x)` token, so each depth position has its own parameters and its own gradient |
| pretrained patch-embed conv | the ImageNet kernel is already 1×1, so it transfers into a `Conv3d(k=1)` with one extra trailing axis |
| data loader | 3D volume pairs; random 3D patches for training, fixed patches for validation, full volumes for test |
| inference | sliding window over Z, Y and X, overlap-averaged, reconstructing the original volume dimensions |

### Geometry constraint

Each patch axis is reduced 16× before the transformer (encoder /4, ART downsample /4) and
restored afterwards, so **every patch dimension must be a positive multiple of 16** and the
patch-embedding stride must be 1. The driver scripts reject violations before allocating a
GPU. (This is also why the original 2D code works at `fineSize=256` but raises
`RuntimeError: Sizes of tensors must match ... Got 76 and 156 in dimension 2` at 624.)

---

## 4. Requirements

The code targets the original ResViT environment and needs **no new packages**:

- Python **3.6**, PyTorch **1.7.1** (CUDA 11.0), torchvision 0.8.2
- numpy 1.19, scipy, scikit-image 0.17, tifffile, Pillow, `ml-collections`, `dominate`, `visdom`

> Python 3.6 is required, not optional: `models/resvit_one.py` uses `async=True` (inherited
> unchanged from upstream ResViT), which is a syntax error on Python ≥ 3.7.

Dataset preparation needs a CZI reader, which is **separate** from the training
environment:

```bash
conda create --name czi3d_env python=3.10 -y
conda activate czi3d_env
pip install bioio bioio-czi tifffile numpy
```

---

## 5. What to download and where to put it

### 5.1 Pretrained ViT weights (required for stage 2)

Stage 2 initialises its transformer from the ImageNet-21k **R50+ViT-B/16** checkpoint —
the same file the original ResViT uses.

- **File:** `R50+ViT-B_16.npz` (461,217,452 bytes, 358 arrays)
- **Source:** the Google ViT model bucket, as specified by the original ResViT and
  TransUNet setup instructions:
  `https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz`
  (the `+` may need to be percent-encoded as `%2B` depending on your client; if the link
  has moved, follow the current instructions in the
  [ResViT repository](https://github.com/icon-lab/ResViT))
- **Place it at exactly this path**, replacing the placeholder file:

```
ResViT-main-3d/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

The path is **relative**, which is why the driver scripts `cd` into `ResViT-main-3d`
before launching Python. Not committed here: 461 MB exceeds GitHub's 100 MB file limit.

Only the transformer half comes from ImageNet. The convolutional half is initialised from
stage 1, which trains from scratch on your own data.

### 5.2 Microscopy data

The 80 source volumes are CZI files from the Allen Institute for Cell Science hiPSC
collection, FBL-mEGFP (fibrillarin) line — see [allencell.org](https://www.allencell.org/).
Not included here (41 GB). Place them in any directory and pass it as `--czi_dir`.

Measured properties of the set used in this work:

| Property | Value |
|---|---|
| shape | `(C=7, Z, Y=624, X=924)`, `SizeS=1`, no T axis |
| dtype | uint16 |
| **Z depth** | **variable: 60 (12 files), 65 (16), 70 (46), 75 (6)** |
| voxel size | 0.1083 × 0.1083 × 0.290 µm (X, Y, Z) |
| anisotropy | Z/XY = **2.68×** |

A loader must not assume a fixed Z.

### 5.3 Trained checkpoints

Not committed (136 MB for stage 1, ~780 MB for stage 2 — both over GitHub's limit).
Share them via an external link if needed.

---

## 6. Usage

### 6.1 Prepare the dataset

```bash
conda activate czi3d_env
python prepare_dataset_3d.py \
  --czi_dir "/path/to/fibrillarin" \
  --out_dir ./data_fibrillarin_3d/native_z \
  --seed 123
```

Inspect one file first without writing anything:

```bash
python prepare_dataset_3d.py --czi_dir "/path/to/fibrillarin" \
  --out_dir ./data_fibrillarin_3d/native_z --limit 1 --dry_run
```

This should report `input = C6 (Bright_100)`, `target = C3 (EGFP)` and a target
contrast around 3.3. A value near 1.2 means the DNA channel is being picked up and the
run should be stopped.

What it does:

- keeps the **native Z** of every volume (nothing cropped, padded, resized or interpolated)
- normalises **per volume and per channel** by clipping to the p0.5–p99.5 percentiles and
  rescaling to uint16
- splits **per source volume**, so no patch from one cell can appear in two splits
  (51 train / 13 val / 16 test with `--seed 123`)
- writes `<case>_X.tif` and `<case>_Y.tif` per cell, plus `norm_params.csv`,
  `split_manifest.csv` and `voxel_spacing.json`

**On intensity semantics:** the stored uint16 is *not* raw camera counts. 0 maps to that
volume's p0.5 and 65535 to its p99.5, with the pair recorded in `norm_params.csv`. The
model therefore predicts **relative, per-volume-normalised** intensity. Recovering
absolute values would require the target volume's percentiles, which do not exist for an
unseen sample, so metrics are computed in the normalised space and never depend on
unavailable target statistics.

### 6.2 Train

Three explicit stages. `STAGE` has no default, and `GPU_AUTHORIZED=1` is required because
the scripts were written for a **shared** multi-user server where a free GPU is not
necessarily an available one.

```bash
GPU_AUTHORIZED=1 GPU=1 CONDA_ENV=resvit_env STAGE=pre bash run_resvit_3d.sh
GPU_AUTHORIZED=1 GPU=1 CONDA_ENV=resvit_env STAGE=ft  bash run_resvit_3d.sh
```

| Stage | Network | Parameters | Starts from | lr | Epochs |
|---|---|---|---|---|---|
| 1 `pre` | `res_cnn` — all 9 ART blocks with `transformer=None`, i.e. a pure 3D CNN | 34.1 M | random init | 2e-4 | 50 + 50 decay |
| 2 `ft` | `resvit` — transformers at ART positions 1 and 6, sharing one encoder | 195.6 M | stage 1 convolutions + ImageNet ViT | 1e-3 | 25 + 25 decay |

Loss: LSGAN + `L1 × 100`, with a 3D PatchGAN discriminator (11.1 M). Unchanged from the
2D recipe.

Configurable without touching code: `PATCH_Z` (48), `PATCH` (128), `BATCH` (1),
`PPV` (8 training patches per volume per epoch), `VPPV` (4 fixed validation patches),
`OVERLAP` (32). Checkpoints and results are keyed by patch size (`z48_xy128`), so runs at
different sizes cannot overwrite each other.

**Why 48 × 128 × 128:** every axis is a multiple of 16; 48 in Z means no volume needs
padding (the shallowest has 60 slices); and it is close to physically isotropic —
48 × 0.29 = 13.9 µm in Z versus 128 × 0.1083 = 13.9 µm in XY. With 51 training volumes and
`PPV=8` an epoch is 408 iterations.

### 6.3 Select a checkpoint, then test

Selection uses **validation only**; the test set is touched once, at the end.

```bash
bash select_best_epoch.sh ./checkpoints_fibrillarin_3d/z48_xy128/fibrillarin3d_BrightFluoro_resvit
GPU_AUTHORIZED=1 GPU=1 CONDA_ENV=resvit_env STAGE=test EPOCH=<best> bash run_resvit_3d.sh
```

`train.py` validates and saves inside the same `if epoch % save_epoch_freq == 0` block, so
every epoch appearing in `log.txt` always has a matching `<epoch>_net_G.pth`; selection can
never name a missing checkpoint, and both scripts verify the file before loading it.

Testing runs the generator in a sliding window at the training patch size (the learned
positional encoding is tied to one input size), averages overlaps, crops off any temporary
boundary padding, and writes:

- `volumes/<case>_fake_B.tif` — predicted fibrillarin volume, original dimensions
- `volumes/<case>_real_B.tif` — ground truth, for comparison
- `metrics.csv` — per-volume MAE, PSNR and SSIM with a **fixed** data range, plus mean ± std
- a central-slice PNG gallery with `index.html`

---

## 7. Repository layout

```
ResViT-main-3d/
├── models/
│   ├── residual_transformers.py   ResViT-3D, ART blocks, 3D embeddings, Res_CNN
│   │                              (Attention/Mlp/Block/Encoder unchanged from 2D)
│   ├── networks.py                define_G/define_D, GANLoss, 3D PatchGAN
│   ├── resvit_one.py              GAN wrapper: losses, optimizers, optimize_parameters
│   ├── transformer_configs.py     ViT-B/16 configuration
│   └── base_model.py              checkpoint save/load
├── data/
│   ├── aligned_dataset.py         3D volume-pair loader (train / val / test behaviours)
│   └── ...
├── options/                       CLI arguments (+ --fineSizeZ, --patches_per_volume,
│                                    --val_patches_per_volume, --overlap)
├── util/                          visualisation, HTML gallery, image pool
├── train.py                       training loop + per-epoch validation  (unchanged from 2D)
├── test.py                        3D sliding-window inference + volume output + metrics
└── model/vit_checkpoint/imagenet21k/   <- put R50+ViT-B_16.npz here
prepare_dataset_3d.py              CZI → uint16 3D TIFF pairs + per-volume split
run_resvit_3d.sh                   stages pre | ft | test, with a shared-GPU guard
test_resvit_3d.sh                  inference only, into its own results directory
select_best_epoch.sh               best validation epoch + checkpoint existence check
```

## 8. Verification

The port was checked incrementally before any training:

- **24 CPU checks** on the architecture: patch grid `(3,8,8)` = 192 tokens at the default
  patch size; the unmodified `Encoder` consuming 3D tokens; `ResViT` and `Res_CNN` forward
  passes; the 3D discriminator; `res_cnn`→`resvit` warm start sharing 140 tensors with 0
  shape mismatches; and per-depth positional-embedding gradients differing, confirming the
  Z positions are independent parameters
- **23 CPU checks** on the loader: exact patch shapes for several configurations;
  voxel-wise input/target alignment after cropping *and* flipping; a deliberately shallow
  volume padding in memory with the source file unchanged on disk; deterministic validation
  patches; full native volumes at test time
- **28 CPU checks** on the sliding window: full axis coverage, exact reconstruction with an
  identity network, crop-back for axes shorter than the patch, overlap averaging, and
  rejection of invalid `--overlap` values
- **GPU validation:** the ImageNet checkpoint loading into the 3D model, and a complete
  training step (generator + discriminator + gradients + both Adam states) measured at
  **6.29 GB** peak allocated for a 48×128×128 patch at batch 1

## 9. Known limitations

- **Relative intensity only** — see §6.1. Absolute fluorescence is not recoverable.
- **51 independent training volumes.** Patches from one volume are correlated; the real
  constraint is the number of cells, not the number of patches.
- **BatchNorm with batch size 1** is noisy. Kept deliberately identical to the 2D recipe so
  training behaviour is comparable rather than silently altered.
- **Isotropic kernels on anisotropic voxels.** Kernels span 2.68× more physical distance in
  Z than in XY. This is deliberate — it matches the elongated axial PSF of a 1.25 NA water
  objective — and anisotropy is handled through the patch aspect ratio instead. No
  resampling is performed, so no data is interpolated into existence.
- **Results are not comparable to earlier 2D numbers**, which were obtained against the
  near-flat `H3342` channel (§2). Predicting an almost constant target is an easier problem
  and yields optimistically high PSNR. A fair 2D baseline would have to be retrained on
  channel 3.
- Inherited upstream quirks, documented rather than silently fixed: `async=True` pinning
  Python 3.6, and unused 2D generators left untouched in `networks.py`.

## 10. Credit

- Original ResViT: Dalmaz, Yurt & Çukur, *ResViT: Residual Vision Transformers for
  Multi-modal Medical Image Synthesis*, IEEE TMI 2022 —
  [icon-lab/ResViT](https://github.com/icon-lab/ResViT)
- Codebase lineage: pix2pix / CycleGAN (Zhu et al.)
- ViT weights: ImageNet-21k R50+ViT-B/16 (Dosovitskiy et al.)
- Data: Allen Institute for Cell Science hiPSC Single-Cell Image Dataset
