# Handwritten Digit Recognition – PyTorch CNN

This project implements a Convolutional Neural Network (CNN) in PyTorch to recognize handwritten digits from the MNIST dataset. It demonstrates a complete **end-to-end deep learning pipeline** for image-based recognition, which is closely related to real-world ID / document digit and text recognition systems.

## Problem formulation

- **Input**: Grayscale images of handwritten digits (28×28 pixels).
- **Classes**: 10 classes (digits 0–9).
- **Task**: Multi-class image classification.

Handwritten digit recognition is a classic computer vision problem and a fundamental building block for **OCR, form processing, and automated document understanding**.

## Approach

- Framework: **PyTorch** and **torchvision** for dataset and transforms.
- Dataset: MNIST downloaded via `torchvision.datasets.MNIST` (70,000 images of handwritten digits).
- Model:
  - A small CNN with convolution, ReLU, max-pooling, and fully connected layers.
  - Trained with **CrossEntropyLoss** and **Adam** optimizer.
- Training:
  - Train/validation split from the official training set.
  - Accuracy tracked on both training and validation data.
- Evaluation:
  - Final test accuracy on held-out MNIST test set.
  - Can be extended with confusion matrix and per-class metrics.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and evaluate the model

```bash
python train_mnist_cnn.py
```

This will:

- Download MNIST automatically if not present.
- Train a CNN for a configurable number of epochs.
- Print training, validation, and test accuracy.

You can tweak hyperparameters (epochs, batch size, learning rate) in `train_mnist_cnn.py`.

## Results

The model was trained for 5 epochs and achieved excellent performance:

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 98.90% |
| Final Test Accuracy | **99.26%** |
| Best Validation Loss | 0.0352 |
| Final Test Loss | 0.0239 |

**Training Summary:**
- Epoch 1: Val Acc = 98.20%
- Epoch 2: Val Acc = 98.72%
- Epoch 3: Val Acc = 98.80% (best validation)
- Epoch 4: Val Acc = 98.72%
- Epoch 5: Val Acc = 98.90% (final best)

The model demonstrates excellent generalization with the test set accuracy (99.26%) exceeding the validation accuracy, indicating a well-regularized model.

## Possible extensions

- Add a notebook to visualize misclassified digits and analyze failure cases.
- Export the trained model and build a small Flask/FastAPI demo for upload-and-predict.
- Try different architectures (deeper CNN, batch normalization, dropout) and compare results.

## Relevance

This project shows:

- Practical **deep learning for vision** using CNNs.
- Clean PyTorch code with clear data pipeline, model definition, and evaluation.
- Experience aligned with **text recognition and ID/document understanding** workflows.
      
