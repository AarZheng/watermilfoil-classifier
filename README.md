# Watermillfoil Image Classifier

A deep learning image classifier for identifying Eurasian watermillfoil (*Myriophyllum spicatum*), an invasive aquatic plant species, using PyTorch and transfer learning.

## Overview

This project implements a binary image classification system to distinguish watermillfoil from other aquatic plants. The classifier uses transfer learning with pre-trained models (EfficientNet-B0 or ResNet50) and achieves ~93-94% validation accuracy.

Eurasian watermillfoil is an invasive species that forms dense mats in water bodies, disrupting ecosystems and interfering with recreation. Automated identification can help with early detection and monitoring efforts.

## Features

- **Transfer Learning**: Uses pre-trained EfficientNet-B0 or ResNet50 models
- **Class Imbalance Handling**: WeightedRandomSampler and class-weighted loss options
- **Data Augmentation**: Comprehensive augmentations for aquatic plant images
- **Mixed Precision Training**: Optional AMP support for faster training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Evaluation**: Confusion matrix, precision, recall, F1-score, and ROC-AUC
- **Inference**: Single image or batch folder prediction
- **Data Collection**: iNaturalist API scraping scripts for dataset creation

## Dataset

The dataset contains images organized in the following structure:

```
dataset/
├── train/
│   ├── millfoil/     # ~3,500+ images
│   └── other/        # ~800 images
├── val/
│   ├── millfoil/     # ~760 images
│   └── other/        # ~100 images
└── test/
    ├── millfoil/     # ~730 images
    └── other/        # ~100 images
```

Images were collected from iNaturalist observations using the provided data collection scripts.

## Requirements

### Python Dependencies

```bash
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
scikit-learn>=1.0.0  # Optional, for detailed metrics
```

### Node.js Dependencies (for data collection)

```bash
sharp>=0.34.3
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm_train
```

2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install torch torchvision pillow scikit-learn
```

4. (Optional) Install Node.js dependencies for data collection:
```bash
npm install
```

## Usage

### Training

Train a model with default settings (EfficientNet-B0, 20 epochs):

```bash
python watermillfoil_classifier.py train \
  --data dataset \
  --epochs 20 \
  --batch-size 32 \
  --img-size 256
```

Train with advanced options:

```bash
python watermillfoil_classifier.py train \
  --data dataset \
  --model efficientnet_b0 \
  --epochs 20 \
  --batch-size 32 \
  --img-size 256 \
  --use-sampler \
  --amp \
  --lr 3e-4 \
  --patience 5
```

Available training options:
- `--model`: Choose `efficientnet_b0` or `resnet50`
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--img-size`: Input image size (default: 256)
- `--use-sampler`: Use WeightedRandomSampler for class imbalance
- `--class-weighted-loss`: Use class-weighted CrossEntropyLoss
- `--amp`: Enable mixed precision training (faster on GPU)
- `--freeze-backbone`: Freeze backbone and only train classifier head
- `--lr`: Learning rate (default: 3e-4)
- `--weight-decay`: Weight decay for AdamW (default: 1e-4)
- `--patience`: Early stopping patience (default: 5)
- `--out-dir`: Output directory for checkpoints (default: runs)

### Evaluation

Evaluate the trained model on the test set:

```bash
python watermillfoil_classifier.py evaluate \
  --data dataset \
  --checkpoint runs/best.pt
```

This will output:
- Overall accuracy, precision, recall, and F1-score
- Per-class classification report
- ROC-AUC score (for binary classification)
- Confusion matrix

### Prediction

Predict on a single image:

```bash
python watermillfoil_classifier.py predict \
  --checkpoint runs/best.pt \
  --image path/to/image.jpg
```

Predict on a folder of images:

```bash
python watermillfoil_classifier.py predict \
  --checkpoint runs/best.pt \
  --folder path/to/images/
```

## Training Results

The model was trained for 20 epochs with the following progression:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1     | 0.4135    | 80.16%    | 0.4747   | 78.37%  | 0.4424 |
| 5     | 0.1396    | 94.90%    | 0.3258   | 90.45%  | 0.4773 |
| 10    | 0.0699    | 97.55%    | 0.2946   | 92.41%  | 0.4828 |
| 15    | 0.0329    | 99.02%    | 0.2423   | 93.67%  | 0.4854 |
| 20    | 0.0217    | 99.25%    | 0.2962   | 92.87%  | 0.4809 |

**Best validation performance**: 93.67% accuracy at epoch 15

## Data Collection Scripts

The `iNaturalist/` directory contains Node.js scripts for collecting training data from the iNaturalist API:

### Scraping Observations

```bash
cd iNaturalist
node get-observations.js
```

This script:
- Fetches observations from iNaturalist for specific taxon IDs
- Extracts high-resolution image URLs
- Saves results to JSON files (e.g., `ewm-images.json`)
- Respects API rate limits (1 second delay between requests)

### Downloading Images

```bash
node download-images.js
```

This script downloads images from the collected URLs.

### Data Splitting

```bash
node move-to-validation.js
```

This script moves a subset of training images to validation set.

## Model Architecture

The classifier uses transfer learning with two available architectures:

### EfficientNet-B0 (Default)
- Pre-trained on ImageNet
- Efficient and accurate
- Input size: 256×256
- Final classifier head replaced for binary classification

### ResNet50
- Pre-trained on ImageNet
- Robust performance
- Input size: 256×256
- Final FC layer replaced for binary classification

## Data Augmentation

Training images undergo the following augmentations:
- Random resized crop (scale: 0.7-1.0)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.2)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random perspective transform (p=0.2)

## Project Structure

```
llm_train/
├── watermillfoil_classifier.py  # Main training/evaluation script
├── dataset/                     # Image dataset
│   ├── train/
│   ├── val/
│   └── test/
├── iNaturalist/                 # Data collection scripts
│   ├── get-observations.js      # Scrape iNaturalist API
│   ├── download-images.js       # Download images
│   └── move-to-validation.js    # Create validation split
├── runs/                        # Training outputs
│   ├── best.pt                  # Best model checkpoint
│   └── classify/                # Training runs
├── yolov8n-cls.pt              # Base YOLOv8 classification model
├── epochs.txt                   # Training epoch logs
└── training_log.txt            # Detailed batch logs
```

## Model Checkpoints

Checkpoints are saved in the `runs/` directory and contain:
- Model state dict
- Optimizer state
- Training metadata (class mappings, model name, image size)
- Best validation metrics

## Troubleshooting

### Truncated Images

The script automatically handles truncated/corrupted images by setting:
```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

### Memory Issues

If you encounter out-of-memory errors:
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--img-size` (try 224)
- Reduce `--num-workers`

### Class Imbalance

The dataset has more "millfoil" images than "other". To handle this:
- Use `--use-sampler` for WeightedRandomSampler
- Or use `--class-weighted-loss` for weighted loss function

## Citation

If you use this code or the methodology, please cite:

```
Watermillfoil Image Classifier
https://github.com/<your-repo>
```

## License

[Add your license information here]

## Acknowledgments

- Images sourced from [iNaturalist](https://www.inaturalist.org/)
- Pre-trained models from torchvision
- Built with PyTorch

## Contact

[Add your contact information here]

