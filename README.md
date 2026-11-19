# U-Net Practice

This repository contains two PyTorch-based medical-image segmentation
exercises:

1. **Nuclei segmentation** using a custom 2D U-Net and the 2018 Data Science
   Bowl dataset.
2. **Hippocampus segmentation** using MONAI, the Medical Segmentation
   Decathlon Task04 dataset, and both a custom and a MONAI U-Net.

The code is primarily a learning and experimentation project. The scripts
train models, display predictions, and compare segmentation metrics such as
accuracy, mean IoU, Dice score, and Hausdorff distance.

## Repository layout

```text
.
├── pyproject.toml
├── uv.lock
├── model_for_nuclei.pt
├── stage1_train/                         # Data Science Bowl training data
├── stage1_test/                          # Data Science Bowl test data
├── U-net Nuclei Segmentation/
│   └── u_net_nuclei_segmentation.py     # 2D RGB nuclei pipeline
└── Segmentation Using MONAI and PyTorch/
    ├── segmentation_with_pytorch_and_monai.py
    ├── monai_tutorial.py
    ├── implemmenting_attention_maps.py
    ├── segmentation_with_pytorch_and_monai.ipynb
    ├── monai_tutorial.ipynb
    └── utils/
        ├── unet.py                       # Custom 2D U-Net
        ├── train.py                      # Training helpers
        ├── metrics.py                    # Dice score helper
        ├── decathlon_dataset.py          # MONAI dataset and transforms
        └── attention_maps.py             # Grad-CAM visualization
```

The datasets, virtual environment, Python cache files, and generated model
files are excluded from version control by `.gitignore`. The repository may
already contain local copies of some of these files for experimentation.

## Requirements and installation

- Python **3.14 or newer**
- PyTorch and torchvision
- NumPy
- scikit-image
- matplotlib
- MONAI
- nibabel
- tqdm (used by the nuclei script; install it separately because it is not
  currently listed in `pyproject.toml`)

The project is configured for [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv pip install tqdm
```

`pyproject.toml` currently selects the CUDA 12.8 PyTorch index. If you need a
CPU-only installation or a different CUDA version, follow the commands on the
[official PyTorch installation page](https://pytorch.org/get-started/locally/)
and adjust the project environment accordingly.

## Project 1: nuclei segmentation

### Dataset

Download the **2018 Data Science Bowl nuclei-segmentation dataset** from
Kaggle. Place the extracted directories in the repository root using this
layout:

```text
stage1_train/
└── <sample-id>/
    ├── images/
    │   └── <sample-id>.png
    └── masks/
        ├── <mask-id>.png
        └── ...

stage1_test/
└── <sample-id>/
    └── images/
        └── <sample-id>.png
```

The script loads RGB images, combines each sample’s individual masks, and
resizes images and masks to `128 × 128`.

### Training and prediction

Run the script from the repository root:

```powershell
uv run python ".\U-net Nuclei Segmentation\u_net_nuclei_segmentation.py"
```

The pipeline:

- uses a reproducible seed of `42`;
- creates a 90/10 training/validation split;
- trains a custom 2D U-Net with `BCEWithLogitsLoss` and Adam;
- trains for at most 25 epochs;
- stops after two validation epochs without improvement;
- uses CUDA when available and otherwise falls back to CPU;
- saves the best checkpoint to `model_for_nuclei.pt`;
- generates binary predictions for training, validation, and test images;
- displays example input, target-mask, and predicted-mask plots.

The test predictions are currently kept in memory and are not exported as a
Kaggle submission file. The script’s constants (`BATCH_SIZE`, `EPOCHS`,
`LEARNING_RATE`, image dimensions, and paths) are defined near the top of
`u_net_nuclei_segmentation.py`.

## Project 2: MONAI hippocampus segmentation

The MONAI project works with **Medical Segmentation Decathlon Task04:
Hippocampus**, a 3D NIfTI dataset. Its data loader uses
`monai.apps.DecathlonDataset` with `download=True`, so the dataset is
downloaded or located under:

```text
Segmentation Using MONAI and PyTorch/
└── utils/
    └── datasets/
        └── Task04_Hippocampus/
```

The preprocessing pipeline loads image/label pairs, ensures channel-first
format, reorients to RAS, resamples spacing, scales image intensity, and
resizes volumes to `(32, 64, 32)`. The custom collate function converts MONAI
MetaTensors into regular tensors for the training code.

Run these scripts from their project directory because their imports and
dataset paths are relative to it:

```powershell
Set-Location ".\Segmentation Using MONAI and PyTorch"

# Train the custom U-Net, save trained_unet.pth, plot predictions,
# and report accuracy, IoU, and Dice metrics.
uv run python .\segmentation_with_pytorch_and_monai.py

# Train MONAI's UNet and calculate the 95th-percentile Hausdorff distance.
uv run python .\monai_tutorial.py

# Load trained_unet.pth and create Grad-CAM attention visualizations.
uv run python .\implemmenting_attention_maps.py
```

The corresponding Jupyter notebooks provide notebook versions of the first
two workflows and can be opened in the same directory after installing the
project dependencies.

### MONAI model details

- The custom model is implemented in `utils/unet.py` and uses convolutional
  blocks, max pooling, transposed-convolution upsampling, and skip
  connections.
- The tutorial model uses `monai.networks.nets.UNet` with 2D spatial
  dimensions, three output classes, channels `(16, 32, 64, 128, 256)`, and
  two residual units per level.
- Training uses cross-entropy loss and Adam with a learning rate of `1e-3`.
- `utils/metrics.py` contains a foreground-class Dice implementation.
- `utils/attention_maps.py` provides Grad-CAM overlays for model inspection.

These scripts are exploratory examples rather than a packaged command-line
application. In particular, batch indexing and plotting assume the dataset
batch contains enough slices, and generated checkpoints must exist before
the attention-map script can run.

## Configuration

The nuclei script’s default settings are:

| Setting | Default |
| --- | ---: |
| Image size | `128 × 128` |
| Image channels | `3` |
| Batch size | `16` |
| Maximum epochs | `25` |
| Learning rate | `0.001` |
| Random seed | `42` |
| Validation split | `10%` |

The MONAI scripts define their batch sizes, number of epochs, dataset task,
and model settings directly in the script files.

## Notes

- Training on CPU can be slow; a CUDA-enabled PyTorch installation is
  recommended for practical experiments.
- Dataset folders and model checkpoints can be large and are intentionally
  ignored by Git.
- The repository includes both 2D and 3D segmentation experiments; the
  nuclei script and the MONAI scripts use different datasets and should be
  run independently.
