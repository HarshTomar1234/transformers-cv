# Variational Autoencoder (VAE)

**Learning Probabilistic Latent Representations**

---

## Table of Contents

1. [Overview](#overview)
2. [Theory & Math](#theory--math)
   - [KL Divergence](#kl-divergence)
   - [ELBO & VAE Loss](#elbo--vae-loss)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Notebook](#notebook)
6. [Architecture Diagrams](#architecture-diagrams)
7. [References](#references)

---

## Overview

Unlike vanilla autoencoders that map inputs to **fixed points** in latent space, VAEs learn a **probabilistic** latent space by encoding inputs to a distribution (mean `μ` and log-variance `log σ²`). This allows:

- **Smooth latent space**: nearby points decode to similar outputs
- **Better generative quality**: samples from the latent space produce coherent, realistic outputs
- **Reparameterization trick**: enables backpropagation through stochastic sampling

### Autoencoders vs VAE

<p align="center">
  <img src="../images/autoencoders vs vae.png" alt="Autoencoders vs VAE comparison" width="700"/>
</p>

---

## Theory & Math

### KL Divergence

KL Divergence measures how one probability distribution diverges from a reference distribution. In VAEs, it regularizes the encoder to produce latent distributions close to a standard normal `N(0, I)`.

<p align="center">
  <img src="../images/kl_div_discrete.png" alt="KL Divergence for Discrete Distributions" width="600"/>
</p>

<p align="center">
  <img src="../images/kl_div_gaussian.png" alt="KL Divergence for Gaussian Distributions" width="600"/>
</p>

### ELBO & VAE Loss

The VAE is trained by maximizing the **Evidence Lower Bound (ELBO)**, which decomposes into:

- **Reconstruction Loss**: How well the decoder reconstructs the input (Binary Cross Entropy)
- **KL Divergence Loss**: How close the learned latent distribution is to the prior `N(0, I)`

```
L_VAE = Reconstruction Loss + KL Divergence
      = BCE(x, x̂) + 0.5 * Σ(μ² + σ² - 1 - log σ²)
```

<p align="center">
  <img src="../images/vae_loss.png" alt="VAE Loss Function" width="600"/>
</p>

<p align="center">
  <img src="../images/vae_loss_derivation.png" alt="VAE Loss Derivation" width="600"/>
</p>

<p align="center">
  <img src="../images/vae_loss_elbo.png" alt="VAE ELBO" width="600"/>
</p>

---

## Architecture

The VAE uses fully-connected layers, encoding to mean and log-variance vectors:

```
Input (784) ──► Encoder ──► μ (128) ──┐
     │                                ├──► z = μ + ε·σ ──► Decoder ──► Output (784)
  Flatten          ──► log σ² (128) ──┘                                    │
  28×28          Reparameterize                                       Sigmoid
                                                                     Reconstruct
```

| Component | Architecture |
|-----------|-------------|
| **Encoder** | Linear(784, 256) → ReLU |
| **μ Head** | Linear(256, 128) |
| **log σ² Head** | Linear(256, 128) |
| **Reparameterize** | z = μ + ε · exp(0.5 · log σ²), where ε ~ N(0, I) |
| **Decoder** | Linear(128, 256) → ReLU → Linear(256, 784) → Sigmoid |

### VAE Architecture Diagram

<p align="center">
  <img src="../images/VAE.png" alt="VAE Architecture Overview" width="600"/>
</p>

---

## Implementation Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | FashionMNIST |
| **Input Dim** | 784 (28×28 flattened) |
| **Hidden Dim** | 256 |
| **Latent Dim** | 128 |
| **Reconstruction Loss** | Binary Cross Entropy (sum reduction) |
| **KL Loss** | 0.5 × Σ(μ² + exp(log σ²) - 1 - log σ²) |
| **Optimizer** | Adam (lr=1e-3) |
| **Batch Size** | 128 |
| **Epochs** | 15 |

### Loss Function (Code)

```python
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
    return recon_loss + kl_loss
```

---

## Notebook

**Notebook**: [VAE_from_scratch.ipynb](VAE_from_scratch.ipynb)

The notebook implements a VAE from scratch in PyTorch and includes:

1. **VAE class** with encoder, reparameterization trick, and decoder
2. **Training** on FashionMNIST for 15 epochs
3. **Reconstruction visualization** — original vs reconstructed images
4. **Random sample generation** — sampling from N(0, I) and decoding
5. **Latent space visualization** — PCA projection of 128D latent space to 2D

---

## Architecture Diagrams

| Resource | File | Description |
|----------|------|-------------|
| Architecture Diagram | [VAE architecture.png](VAE%20architecture.png) | Detailed VAE architecture overview |
| Editable Source | [VAE architecture overview.excalidraw](VAE%20architecture%20overview.excalidraw) | Excalidraw source file |
| Research Paper | [VAE paper.pdf](VAE%20paper.pdf) | Original VAE paper (Kingma & Welling, 2013) |

---

## References

| Resource | Link |
|----------|------|
| Auto-Encoding Variational Bayes (Kingma & Welling, 2013) | [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114) |
| Tutorial on VAEs (Doersch, 2016) | [arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908) |
| Understanding VAEs | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/) |
| PyTorch Documentation | [pytorch.org](https://pytorch.org/) |
