# AutoEncoders

**Learning Representation Through Encoding and Decoding**

---

## Table of Contents

1. [Overview](#overview)
2. [Vanilla Autoencoder](#vanilla-autoencoder)
   - [Architecture](#architecture)
   - [Notebook](#notebook)
3. [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
4. [Folder Structure](#folder-structure)
5. [Implementation Status](#implementation-status)
6. [References](#references)

---

## Overview

Autoencoders are a family of unsupervised neural networks that learn efficient, compressed representations of input data. They consist of two main components:

- **Encoder**: Maps the input to a lower-dimensional latent space
- **Decoder**: Reconstructs the input from the latent representation

The model is trained to minimize the reconstruction error between the original input and the decoded output, forcing the network to learn meaningful features in the latent space.

### Key Concepts

- **Latent Space**: The compressed representation learned by the encoder — captures the most important features of the data
- **Reconstruction Loss**: Measures how well the decoder can reconstruct the original input (typically MSE for continuous data)
- **Dimensionality Reduction**: Similar in spirit to PCA, but capable of learning non-linear transformations
- **Generative Capability**: Once trained, the decoder can generate new samples by sampling from the latent space

---

## Vanilla Autoencoder

### Architecture

The vanilla autoencoder uses fully-connected (linear) layers with ReLU activations:

```
Input (784) ──► Encoder ──► Latent Space (32) ──► Decoder ──► Output (784)
     │                           │                              │
     │                           │                              │
  Flatten                   Bottleneck                     Sigmoid
  28×28                    Compressed                     Reconstruct
```

| Component | Architecture |
|-----------|-------------|
| **Encoder** | Linear(784, 256) → ReLU → Linear(256, 32) → ReLU |
| **Decoder** | Linear(32, 256) → ReLU → Linear(256, 784) → Sigmoid |
| **Loss** | MSE (Mean Squared Error) |
| **Optimizer** | Adam (lr=1e-3) |
| **Latent Dim** | 32 |
| **Hidden Dim** | 256 |

### Notebook

**Notebook**: [Vanilla Autoencoder/AutoEncoders_coding.ipynb](Vanilla%20Autoencoder/AutoEncoders_coding.ipynb)

The notebook implements the vanilla autoencoder from scratch in PyTorch on two datasets:

#### MNIST (Handwritten Digits)

- Training for **15 epochs** with batch size 128
- Final loss: **0.010073**
- Reconstruction visualization (original vs. reconstructed)
- Latent space visualization using PCA (32D → 2D projection)
- Random sample generation from latent space

#### FashionMNIST (Clothing Items)

- Training for **20 epochs** with batch size 128
- Final loss: **0.010340**
- Reconstruction visualization (original vs. reconstructed)
- Latent space visualization using PCA (32D → 2D projection)

### Architecture Diagram

| Resource | File | Description |
|----------|------|-------------|
| Overview Diagram | [Autoencoders overview.png](Vanilla%20Autoencoder/Autoencoders%20overview.png) | Visual overview of autoencoder architecture |
| Editable Source | [Autoencoders.excalidraw](Vanilla%20Autoencoder/Autoencoders.excalidraw) | Excalidraw source file for the architecture diagram |

### Key Observations

- The vanilla autoencoder successfully compresses 784-dimensional inputs into 32 dimensions and reconstructs them with low MSE
- PCA projections of the latent space show some clustering by digit/class, but with significant overlap
- Random samples from the latent space produce noisy, less coherent reconstructions — a known limitation of vanilla autoencoders that motivates the move to VAEs

---

## Variational Autoencoder (VAE)

> **Coming Soon** — VAE implementation with reparameterization trick, KL divergence loss, and improved generative capabilities.

VAEs address the limitations of vanilla autoencoders by:
- Learning a **probabilistic** latent space (encoding to a distribution rather than a fixed point)
- Using the **reparameterization trick** for differentiable sampling
- Adding **KL divergence** to the loss, encouraging a smooth, continuous latent space
- Producing **higher quality** generated samples

---

## Folder Structure

```
AutoEncoders/
├── README.md                                    # This file
├── Vanilla Autoencoder/
│   ├── AutoEncoders_coding.ipynb                # From-scratch implementation
│   ├── Autoencoders overview.png                # Architecture diagram
│   └── Autoencoders.excalidraw                  # Editable diagram source
└── VAE/                                         # Coming soon
    ├── VAE_coding.ipynb                         # (Planned)
    └── VAE_architecture.excalidraw              # (Planned)
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Vanilla Autoencoder (MNIST) | ✅ Complete |
| Vanilla Autoencoder (FashionMNIST) | ✅ Complete |
| Architecture Diagram | ✅ Complete (Excalidraw + PNG) |
| Latent Space Visualization | ✅ Complete |
| VAE Implementation | 🔲 Planned |
| VAE Architecture Diagram | 🔲 Planned |

---

## References

| Resource | Link |
|----------|------|
| Autoencoder (Wikipedia) | [en.wikipedia.org/wiki/Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) |
| VAE Paper (Kingma & Welling, 2013) | [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114) |
| PyTorch Documentation | [pytorch.org](https://pytorch.org/) |
| Understanding Autoencoders | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/) |

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
