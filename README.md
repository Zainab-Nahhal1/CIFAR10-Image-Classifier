# CIFAR-10 CNN Classifier
**Project:** cifar10-cnn-classifier

## Overview
This repository contains a compact Convolutional Neural Network (CNN) implementation
to classify images from the CIFAR-10 dataset. The goal is to provide an easy-to-run
project skeleton (data loading, model definition, training loop, evaluation and plotting),
suitable as a learning resource or a starting point for experiments.

## Contents
- `source/main.py` — main training and evaluation code (functions: `load_data`, `build_model`, `train`, `evaluate`).
- `sample/` — a placeholder directory for sample images and generated plots.
- `tests/test_model.py` — basic tests that ensure model and data shapes are correct.
- `Makefile` — convenience commands (`train`, `test`, `clean`).
- `requirements.txt` — runtime dependencies.

## Dataset
CIFAR-10 is a standard benchmark dataset of 60,000 32x32 colour images in 10 classes,
with 50,000 training images and 10,000 test images.

The script uses a Google Cloud mirror to download CIFAR-10 (faster and more reliable):
```
https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz
```

## Model architecture
The provided CNN uses a simple but effective stack of convolutional and pooling layers:
- Conv2D(32) -> MaxPool
- Conv2D(64) -> MaxPool
- Conv2D(64) -> Flatten -> Dense(128) -> Dropout(0.4) -> Dense(10)

The model is compiled with Adam optimizer and `SparseCategoricalCrossentropy(from_logits=True)` loss.

## Training
Run training locally or in Colab:
```bash
python -m source.main --epochs 10 --batch-size 64
```

The training routine:
- Saves a best-model checkpoint (`cifar10_best.h5`).
- Produces plots saved under `sample/plots/`:
  - `accuracy_curve.png`
  - `loss_curve.png`
  - `confusion_matrix.png`

## Evaluation & Metrics
After training, the script computes predictions on the test set and prints:
- Classification report (precision, recall, f1-score for each class)
- Confusion matrix (saved as a heatmap)

## Tests
Run tests with:
```bash
pytest -q
```
(or `python -m unittest discover -v`)

## Future improvements
- Add data augmentation using `ImageDataGenerator` or `tf.image` transforms.
- Experiment with transfer learning (ResNet, EfficientNet).
- Tune hyperparameters: optimizer, learning-rate schedule, batch size, dropout, weight decay.
- Convert to TensorFlow `tf.data` pipeline for better performance.

## Notes
- This repository ships code that will download the dataset on first run and cache it in the Keras cache directory `~/.keras/datasets/`.
- Training on CPU may be slow; use GPU (Colab or a local machine with CUDA) for faster results.