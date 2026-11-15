# U-Net Nuclei Segmentation

A small PyTorch project for segmenting cell nuclei in microscopy images. It
implements a U-Net model and trains it on the **2018 Data Science Bowl**
nuclei-segmentation dataset.

The script:

- loads training and test images from the Kaggle dataset layout;
- resizes images and masks to `128 × 128`;
- creates a reproducible 90/10 training/validation split;
- trains with `BCEWithLogitsLoss` and the Adam optimizer;
- uses early stopping and saves the best model to `model_for_nuclei.pt`;
- displays example input, ground-truth mask, and predicted mask images.

## Project layout

```text
.
├── u_net_pytorch.py       # Model, data loading, training, and prediction
├── model_for_nuclei.pt    # Saved model checkpoint
├── pyproject.toml         # Python project and dependency configuration
├── stage1_train/          # Training images and masks
└── stage1_test/           # Test images
```

Each dataset sample should have this structure:

```text
stage1_train/
└── <sample-id>/
    ├── images/
    │   └── <sample-id>.png
    └── masks/
        ├── <mask-id>.png
        └── ...
```

The test set uses the same structure but does not contain `masks/`.

## Requirements

- Python 3.14 or newer
- PyTorch
- NumPy
- scikit-image
- matplotlib
- tqdm

The project is configured for [uv](https://docs.astral.sh/uv/), but the
dependencies can also be installed with another Python environment manager.

## Setup with uv

From the project directory:

```powershell
uv sync
```

Install PyTorch separately in the environment. Choose the CPU or CUDA command
recommended by the [official PyTorch installation
page](https://pytorch.org/get-started/locally/).

Make sure the dataset is available beside the script:

```text
D:\New folder\AI\U_net_practice\
├── stage1_train\
└── stage1_test\
```

Download the **2018 Data Science Bowl** dataset from Kaggle and extract its
`stage1_train` and `stage1_test` directories into the project directory.

## Run training

```powershell
uv run python u_net_pytorch.py
```

The program automatically uses a CUDA GPU when one is available; otherwise it
uses the CPU. Training progress is printed for each epoch. The best checkpoint
is written to:

```text
model_for_nuclei.pt
```

After training, two figures are shown: one for a training image and one for a
validation image. Each figure contains the input image, the target mask, and
the predicted binary mask.

## Configuration

The main training settings are defined near the top of
`u_net_pytorch.py`:

| Setting | Default |
| --- | ---: |
| Image size | `128 × 128` |
| Batch size | `16` |
| Maximum epochs | `25` |
| Learning rate | `0.001` |
| Random seed | `42` |
| Validation split | `10%` |

Change these constants before running the script if you want to experiment
with a different configuration.

## Notes

- The current script keeps test predictions in memory for demonstration; it
  does not export a submission file.
- Large dataset folders and model checkpoints may require substantial disk
  space and are best kept outside version control.