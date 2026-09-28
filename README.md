# SADRFNet

PyTorch implementation of **SADRFNet (Scene-Adaptive Dilated Reparameterization Fusion Network)** for remote sensing scene classification.

## Project Structure

```text
SADRFNet/
├── models/
│   └── sadrfnet_lg.py
├── dataset_builder.py
├── main_lg_branch_weights.py
├── train.py
├── test.py
└── README.md
```

## Files

- **`models/sadrfnet_lg.py`**  
  Implements the SADRFNet model, including the dilated reparameterization context branch, scene-adaptive local-global fusion, progressive multi-stage fusion, and final classification module. :chatgpt-content-reference{index="0"}

- **`dataset_builder.py`**  
  Loads remote sensing scene datasets, generates stratified train/validation/test splits, applies image preprocessing and augmentation, and builds PyTorch DataLoaders. It currently supports **AID**, **NWPU-RESISC45**, and **UC Merced Land Use**. :chatgpt-content-reference{index="1"}

- **`main_lg_branch_weights.py`**  
  Main experiment script for the current SADRFNet local-global model. It contains the major training, testing, model configuration, optimizer, checkpoint, evaluation, and local-global branch-weight analysis functions. :chatgpt-content-reference{index="2"}

- **`train.py`**  
  Training script for SADRFNet, including model construction, optimizer and scheduler configuration, checkpoint saving, and training/evaluation loops. :chatgpt-content-reference{index="3"}

- **`test.py`**  
  Independent testing script for loading trained checkpoints and evaluating classification performance, including OA, AA, Macro-F1, Kappa, confusion matrix, and per-class accuracy. :chatgpt-content-reference{index="4"}

## Datasets

The code supports the following remote sensing scene classification datasets:

- UC Merced Land Use
- AID
- NWPU-RESISC45

Please organize the datasets as:

```text
datasets/
├── UCMerced_LandUse/
├── AID/
└── NWPU-RESISC45/
```

Each dataset directory should contain one subfolder for each scene category.

## Requirements

Main dependencies:

```text
Python
PyTorch
torchvision
NumPy
Pillow
tqdm
```

## Usage

The recommended entry point for the current SADRFNet implementation is:

```bash
python main_lg_branch_weights.py
```

Experiment settings such as dataset, training ratio, epochs, batch size, learning rate, and model configuration can be modified in the `USER CONFIG` section of `main_lg_branch_weights.py`.

## Citation

Citation information will be updated after the paper is published.
