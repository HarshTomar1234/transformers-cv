# AutoEncoders

**Learning Representation Through Encoding and Decoding**

---

## Table of Contents

1. [Overview](#overview)
2. [Visual Overview](#visual-overview)
3. [Vanilla Autoencoder](#vanilla-autoencoder)
4. [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
5. [VQ-VAE](#vq-vae)
6. [RVQ (Residual Vector Quantization)](#rvq-residual-vector-quantization)
7. [Folder Structure](#folder-structure)
8. [Implementation Status](#implementation-status)
9. [References](#references)

---

## Overview

Autoencoders are a family of unsupervised neural networks that learn efficient, compressed representations of input data. They consist of two main components:

- **Encoder**: Maps the input to a lower-dimensional latent space
- **Decoder**: Reconstructs the input from the latent representation

The model is trained to minimize the reconstruction error, forcing the network to learn meaningful features in the latent space.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Latent Space** | Compressed representation learned by the encoder — captures the most important features |
| **Reconstruction Loss** | Measures how well the decoder reconstructs the original input |
| **Dimensionality Reduction** | Similar to PCA, but capable of learning non-linear transformations |
| **Generative Capability** | Decoder can generate new samples by sampling from the latent space |

---

## Visual Overview

### Autoencoders vs VAE

<p align="center">
  <img src="images/autoencoders vs vae.png" alt="Autoencoders vs VAE" width="700"/>
</p>

### KL Divergence

KL Divergence measures how one probability distribution diverges from a reference distribution. It's a key component of the VAE loss function.

<p align="center">
  <img src="images/kl_div_discrete.png" alt="KL Divergence — Discrete Distributions" width="600"/>
</p>

<p align="center">
  <img src="images/kl_div_gaussian.png" alt="KL Divergence — Gaussian Distributions" width="600"/>
</p>

### VAE Loss Function

<p align="center">
  <img src="images/vae_loss.png" alt="VAE Loss" width="600"/>
</p>

<p align="center">
  <img src="images/vae_loss_derivation.png" alt="VAE Loss Derivation" width="600"/>
</p>

<p align="center">
  <img src="images/vae_loss_elbo.png" alt="VAE ELBO" width="600"/>
</p>

---

## Vanilla Autoencoder

The vanilla autoencoder uses fully-connected linear layers with deterministic encoding:

```
Input (784) ──► Encoder ──► Latent (32) ──► Decoder ──► Output (784)
```

| Component | Architecture |
|-----------|-------------|
| **Encoder** | Linear(784, 256) → ReLU → Linear(256, 32) → ReLU |
| **Decoder** | Linear(32, 256) → ReLU → Linear(256, 784) → Sigmoid |
| **Loss** | MSE (Mean Squared Error) |
| **Optimizer** | Adam (lr=1e-3) |

### Datasets & Results

| Dataset | Epochs | Final Loss |
|---------|--------|------------|
| MNIST | 15 | 0.010073 |
| FashionMNIST | 20 | 0.010340 |

### What's Included

- **Notebook**: [AutoEncoders_coding.ipynb](Vanilla%20Autoencoder/AutoEncoders_coding.ipynb) — From-scratch implementation
- **Architecture Diagram**: [Autoencoders overview.png](Vanilla%20Autoencoder/Autoencoders%20overview.png)
- **Editable Diagram**: [Autoencoders.excalidraw](Vanilla%20Autoencoder/Autoencoders.excalidraw)
- **Paper**: [AutoEncoders paper.pdf](Vanilla%20Autoencoder/AutoEncoders%20paper.pdf)

### Key Observations

- Successfully compresses 784D inputs to 32D and reconstructs with low MSE
- PCA projections show some class clustering, but with significant overlap
- Random latent samples produce noisy outputs — motivating the move to VAEs

📁 **[Full Vanilla AE README →](Vanilla%20Autoencoder/)** *(inside subfolder)*

---

## Variational Autoencoder (VAE)

VAEs address the limitations of vanilla autoencoders by learning a **probabilistic** latent space:

```
Input (784) ──► Encoder ──► μ (128) ──┐
                                      ├──► z = μ + ε·σ ──► Decoder ──► Output (784)
                  ──► log σ² (128) ──┘
                  Reparameterization Trick
```

| Component | Architecture |
|-----------|-------------|
| **Encoder** | Linear(784, 256) → ReLU |
| **μ Head** | Linear(256, 128) |
| **log σ² Head** | Linear(256, 128) |
| **Reparameterize** | z = μ + ε · exp(0.5 · log σ²) |
| **Decoder** | Linear(128, 256) → ReLU → Linear(256, 784) → Sigmoid |
| **Loss** | BCE (reconstruction) + KL Divergence |
| **Optimizer** | Adam (lr=1e-3) |

### VAE Architecture

<p align="center">
  <img src="images/VAE.png" alt="VAE Architecture" width="600"/>
</p>

### Dataset & Training

| Parameter | Value |
|-----------|-------|
| **Dataset** | FashionMNIST |
| **Latent Dim** | 128 |
| **Epochs** | 15 |
| **Batch Size** | 128 |

### What's Included

- **Notebook**: [VAE_from_scratch.ipynb](VAE/VAE_from_scratch.ipynb) — Full VAE implementation
- **Architecture Diagram**: [VAE architecture.png](VAE/VAE%20architecture.png)
- **Editable Diagram**: [VAE architecture overview.excalidraw](VAE/VAE%20architecture%20overview.excalidraw)
- **Paper**: [VAE paper.pdf](VAE/VAE%20paper.pdf)
- **README**: [Full VAE README →](VAE/README.md) — Detailed theory, math, and implementation

### Key Improvements over Vanilla AE

- Learns a **probabilistic** latent space (distribution, not fixed points)
- **Reparameterization trick** enables differentiable sampling
- **KL divergence** encourages smooth, continuous latent space
- **Higher quality** generated samples from random latent vectors

📁 **[Full VAE README →](VAE/README.md)** *(with detailed theory & math)*

---

## VQ-VAE

> **Coming Soon** — Vector Quantized VAE implementation with discrete latent representations.

VQ-VAE replaces continuous latent variables with **discrete codebook** vectors, learning a finite set of embedding vectors that inputs can be mapped to.

---

## RVQ (Residual Vector Quantization)

> **Coming Soon** — Residual Vector Quantization for hierarchical discrete representations.

RVQ extends VQ-VAE by using **multiple stages of quantization**, where each stage encodes the residual error from the previous stage, enabling more expressive discrete representations.

---

## Folder Structure

```
AutoEncoders/
├── README.md                                          # This file
├── images/                                            # All visual resources
│   ├── VAE.png                                        # VAE architecture diagram
│   ├── autoencoders vs vae.png                        # AE vs VAE comparison
│   ├── kl_div_discrete.png                            # KL divergence (discrete)
│   ├── kl_div_gaussian.png                            # KL divergence (Gaussian)
│   ├── vae_loss.png                                   # VAE loss function
│   ├── vae_loss_derivation.png                        # VAE loss derivation
│   └── vae_loss_elbo.png                              # VAE ELBO
├── Vanilla Autoencoder/
│   ├── AutoEncoders_coding.ipynb                      # From-scratch implementation
│   ├── Autoencoders overview.png                      # Architecture diagram
│   ├── Autoencoders.excalidraw                        # Editable diagram source
│   └── AutoEncoders paper.pdf                         # Reference paper
├── VAE/
│   ├── README.md                                      # Detailed VAE README
│   ├── VAE_from_scratch.ipynb                         # From-scratch implementation
│   ├── VAE architecture.png                           # Architecture diagram
│   ├── VAE architecture overview.excalidraw           # Editable diagram source
│   └── VAE paper.pdf                                  # Original VAE paper
├── VQ-VAE/                                            # Coming soon
└── RVQ/                                               # Coming soon
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Vanilla Autoencoder (MNIST) | ✅ Complete |
| Vanilla Autoencoder (FashionMNIST) | ✅ Complete |
| Vanilla AE Architecture Diagram | ✅ Complete |
| Vanilla AE Latent Space Visualization | ✅ Complete |
| VAE Implementation (FashionMNIST) | ✅ Complete |
| VAE Architecture Diagram | ✅ Complete |
| VAE Theory Images (KL, ELBO, Loss) | ✅ Complete |
| VQ-VAE Implementation | 🔲 Planned |
| RVQ Implementation | 🔲 Planned |

---

## References

| Resource | Link |
|----------|------|
| Autoencoder (Wikipedia) | [en.wikipedia.org/wiki/Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) |
| VAE Paper (Kingma & Welling, 2013) | [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114) |
| Tutorial on VAEs (Doersch, 2016) | [arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908) |
| Understanding Autoencoders & VAEs | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/) |
| PyTorch Documentation | [pytorch.org](https://pytorch.org/) |

---

**Note**: This is part of the [transformers-CV](../) repository — computer vision architectures implemented from scratch.
