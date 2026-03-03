# Vanilla Autoencoder

**Deterministic Encoding for Unsupervised Representation Learning**

---

## Overview

A vanilla autoencoder learns to compress high-dimensional input into a fixed lower-dimensional latent vector and reconstruct the original input from it. The encoder and decoder are trained jointly by minimizing reconstruction error.

---

## Architecture

```
Input (784) ──► Encoder ──► Latent (32) ──► Decoder ──► Output (784)
```

| Component | Architecture |
|-----------|-------------|
| **Encoder** | Linear(784, 256) → ReLU → Linear(256, 32) → ReLU |
| **Decoder** | Linear(32, 256) → ReLU → Linear(256, 784) → Sigmoid |
| **Loss** | MSE (Mean Squared Error) |
| **Optimizer** | Adam (lr=1e-3) |

---

## Datasets & Results

| Dataset | Epochs | Final Loss |
|---------|--------|------------|
| MNIST (Handwritten Digits) | 15 | 0.010073 |
| FashionMNIST (Clothing Items) | 20 | 0.010340 |

---

## Notebook Contents

**Notebook**: [AutoEncoders_coding.ipynb](AutoEncoders_coding.ipynb)

The notebook implements the vanilla autoencoder from scratch in PyTorch and includes:

1. **Model definition** — Encoder/decoder with `nn.Linear` layers
2. **Training on MNIST** — 15 epochs, reconstruction visualization, latent space PCA (1D, 2D, 3D), random sample generation
3. **Training on FashionMNIST** — 20 epochs, reconstruction visualization, latent space PCA

### Key Observations

- Successfully compresses 784-dimensional inputs into 32 dimensions with low reconstruction error
- PCA projections of the latent space show partial class clustering, but with significant overlap
- Random samples from the latent space produce noisy, less coherent outputs — a known limitation that motivates the move to VAEs

---

## Files

| File | Description |
|------|-------------|
| [AutoEncoders_coding.ipynb](AutoEncoders_coding.ipynb) | From-scratch implementation notebook |
| [Autoencoders overview.png](Autoencoders%20overview.png) | Architecture overview diagram |
| [Autoencoders.excalidraw](Autoencoders.excalidraw) | Editable Excalidraw source |
| [AutoEncoders paper.pdf](AutoEncoders%20paper.pdf) | Reference paper |

---

## References

| Resource | Link |
|----------|------|
| Autoencoder (Wikipedia) | [en.wikipedia.org/wiki/Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) |
| PyTorch Documentation | [pytorch.org](https://pytorch.org/) |
