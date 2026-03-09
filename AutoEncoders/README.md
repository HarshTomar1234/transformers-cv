# AutoEncoders

**Learning Representation Through Encoding and Decoding**

---

## Overview

Autoencoders are a family of unsupervised neural networks that learn efficient, compressed representations of input data. They consist of two main components:

- **Encoder**: Maps the input to a lower-dimensional latent space
- **Decoder**: Reconstructs the input from the latent representation

This folder contains from-scratch PyTorch implementations of multiple autoencoder variants, progressing from deterministic to probabilistic to discrete representations.

---

## Visual Overview

### Autoencoders vs VAE

<p align="center">
  <img src="images/autoencoders vs vae.png" alt="Autoencoders vs VAE" width="700"/>
</p>

### KL Divergence

<p align="center">
  <img src="images/kl_div_discrete.png" alt="KL Divergence — Discrete Distributions" width="600"/>
</p>

<p align="center">
  <img src="images/kl_div_gaussian.png" alt="KL Divergence — Gaussian Distributions" width="600"/>
</p>

### VAE Loss (ELBO)

<p align="center">
  <img src="images/vae_loss.png" alt="VAE Loss" width="600"/>
</p>

<p align="center">
  <img src="images/vae_loss_elbo.png" alt="VAE ELBO" width="600"/>
</p>

<p align="center">
  <img src="images/vae_loss_derivation.png" alt="VAE Loss Derivation" width="600"/>
</p>

### VQ-VAE Architecture

<p align="center">
  <img src="images/VQ-VAE.png" alt="VQ-VAE Overview" width="600"/>
</p>

<p align="center">
  <img src="images/VQ-VAE architecture.png" alt="VQ-VAE Architecture Diagram" width="700"/>
</p>

### Vector Quantization (K-Means Intuition)

<p align="center">
  <img src="images/K-means.png" alt="K-Means Clustering Intuition" width="600"/>
</p>

### Bit Depth & Codebook Capacity

<p align="center">
  <img src="images/bit-depth.png" alt="Bit Depth and Codebook Capacity" width="600"/>
</p>

---

## Implementations

### [Vanilla Autoencoder](Vanilla%20Autoencoder/)

Deterministic autoencoder with fully-connected layers. Encodes 784D inputs to a 32D latent space using MSE loss. Trained on MNIST and FashionMNIST.

### [Variational Autoencoder (VAE)](VAE/)

Probabilistic autoencoder that learns a latent distribution (mean and log-variance) using the reparameterization trick. Uses BCE + KL divergence loss. Trained on FashionMNIST with a 128D latent space.

<p align="center">
  <img src="images/VAE.png" alt="VAE Architecture" width="500"/>
</p>

### [Vector Quantized VAE (VQ-VAE)](VQVAE/)

VQ-VAE replaces the continuous latent space with a **discrete codebook** of learned embedding vectors. The encoder output is quantized to the nearest codebook entry via vector quantization. Uses the **straight-through estimator** for gradient flow and a combined reconstruction + codebook + commitment loss. Both linear and convolutional variants are implemented, trained on FashionMNIST.

<p align="center">
  <img src="images/VQ-VAE.png" alt="VQ-VAE Architecture" width="500"/>
</p>

### RVQ *(Coming Soon)*

Residual Vector Quantization — multi-stage quantization for hierarchical discrete representations.

---

## Folder Structure

```
AutoEncoders/
├── README.md
├── images/
│   ├── VAE.png
│   ├── VQ-VAE.png
│   ├── VQ-VAE architecture.png
│   ├── K-means.png
│   ├── bit-depth.png
│   ├── autoencoders vs vae.png
│   ├── kl_div_discrete.png
│   ├── kl_div_gaussian.png
│   ├── vae_loss.png
│   ├── vae_loss_derivation.png
│   └── vae_loss_elbo.png
├── Vanilla Autoencoder/
│   ├── README.md
│   ├── AutoEncoders_coding.ipynb
│   ├── Autoencoders overview.png
│   ├── Autoencoders.excalidraw
│   └── AutoEncoders paper.pdf
├── VAE/
│   ├── README.md
│   ├── VAE_from_scratch.ipynb
│   ├── VAE architecture.png
│   ├── VAE architecture overview.excalidraw
│   └── VAE paper.pdf
├── VQVAE/
│   ├── README.md
│   ├── Vector_Quantized_Variational_AutoEncoders.ipynb
│   ├── models.py
│   ├── utils.py
│   └── VQVAE paper.pdf
└── RVQ/                             (Planned)
```

---

## References

| Resource | Link |
|----------|------|
| Autoencoder (Wikipedia) | [en.wikipedia.org/wiki/Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) |
| VAE Paper (Kingma & Welling, 2013) | [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114) |
| VQ-VAE Paper (van den Oord et al., 2017) | [arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937) |
| Tutorial on VAEs (Doersch, 2016) | [arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908) |
| Understanding VAEs | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/) |
| PyTorch Documentation | [pytorch.org](https://pytorch.org/) |
