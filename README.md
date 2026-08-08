# Shoplifting Detection — Video Classification

A video classification pipeline that distinguishes normal shopping behavior from shoplifting/theft acts using a from-scratch temporal deep learning architecture.

## Overview

The model classifies short surveillance video clips into two categories — **shoplifter** vs **non shoplifter** — using a 3D Convolutional Neural Network trained entirely from scratch (no pretrained weights).

## Dataset

- **Source**: Shoplifting Dataset (Kaggle), organized into `shop lifters` and `non shop lifters` folders.
- **Raw size**: 855 video clips (704×576 resolution, ~25 fps).
- **Cleaning**: Identified and removed 225 duplicate clips (files with a `_1` filename suffix), leaving **630 unique clips** — 319 shoplifters / 311 non-shoplifters, a near-balanced set.

## Pipeline

1. **Exploratory Data Analysis** — class distribution, clip duration and frame-count distributions, resolution check, and visual inspection of sample frames and statistical outliers.
2. **Duplicate Removal** — regex-based detection of `_1`-suffixed duplicate files, dropped prior to splitting to avoid train/test leakage.
3. **Frame Sampling Strategy** — a hybrid sampler that combines:
   - **Uniform sampling** across the full clip duration (guarantees even temporal coverage regardless of clip length).
   - **Motion-aware sampling**, using frame-to-frame pixel-difference scores to bias frame selection toward high-motion moments, reducing the risk of missing brief shoplifting actions.
   - Short clips (fewer frames than the target `T`) are handled by taking every frame and padding with repeated final frames.
4. **PyTorch Dataset** — returns clip tensors of shape `(T, C, H, W)`. Random flip, crop, and color-jitter parameters are sampled **once per clip** and applied identically to every sampled frame, preserving temporal consistency of the augmentation.
5. **Model — 3D CNN (from scratch)** — a 4-block Conv3D/BatchNorm3D/ReLU/MaxPool3D architecture with global average pooling and a fully connected classifier head. No pretrained backbone is used; weights are initialized with Kaiming initialization.
6. **Training** — stratified 70/15/15 train/val/test split, class-weighted cross-entropy loss, Adam optimizer with weight decay, `ReduceLROnPlateau` learning-rate scheduling, and early stopping based on validation F1-score.
7. **Evaluation** — accuracy, precision, recall, and F1-score computed on the held-out test set, along with a confusion matrix and training curves.

## Requirements

- Python 3.x
- PyTorch, torchvision
- OpenCV (`opencv-python`)
- pandas, numpy, matplotlib, seaborn
- scikit-learn

## Notes

- The model is trained **entirely from scratch** — no transfer learning.
- The frame sampling strategy was specifically designed to reduce the chance of missing short, localized shoplifting actions within longer clips, in addition to satisfying the requirement of even coverage across the full video duration.
