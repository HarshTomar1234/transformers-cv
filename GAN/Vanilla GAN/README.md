# Vanilla GAN — Introduction to Generative Adversarial Networks

A from-scratch PyTorch implementation of Generative Adversarial Networks on MNIST, covering the complete mathematical derivation and three progressive GAN variants: **Unconditional**, **Conditional**, and **Convolutional**.

---

## Architecture Overview

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/GAN%20architecture.png?raw=true" alt="GAN Architecture" width="700"/>

A GAN consists of two competing networks:
- **Generator** — Takes random noise $z$ and produces fake images $G(z)$
- **Discriminator** — Takes real and generated images and classifies them as real or fake

The two networks play a minimax game described by the **GAN objective**:

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{data}} [\log D(x)] + \mathbb{E}_{z\sim p_z} [\log(1 - D(G(z)))]$$

---

## Mathematical Derivation

The notebook provides a **complete, step-by-step derivation** of the GAN loss function:

| Topic | Description |
|-------|-------------|
| **BCE Loss Formulation** | Deriving discriminator loss from Binary Cross-Entropy for real and fake images |
| **Continuous Extension** | Moving from discrete labels to continuous probability distributions |
| **Optimal Discriminator** | Proving $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$ |
| **Generator Loss** | Showing how the generator minimizes discriminator confidence |
| **Jensen-Shannon Divergence** | Proving the GAN objective equals $-2\log 2 + 2 \cdot D_{JS}(p_{data} \| p_g)$ |

---

## Implementations

### 1. Unconditional Linear GAN
A simple fully-connected GAN that generates random MNIST digits from gaussian noise.

- **Generator**: Linear layers (`100 → 256 → 512 → 1024 → 784`) with LeakyReLU + Dropout, ending with Tanh
- **Discriminator**: Linear layers (`784 → 512 → 256 → 1`) with LeakyReLU + Dropout
- **Loss**: `BCEWithLogitsLoss` with label smoothing

### 2. Conditional GAN (cGAN)

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/Conditional%20GAN.png?raw=true" alt="Conditional GAN" width="700"/>

Adds **class conditioning** via learned embeddings — enabling controlled generation of specific digits (0–9).

- Embedding matrices in both Generator and Discriminator (10 embeddings, dim=16)
- Digit label embeddings are concatenated with noise/image features
- Discriminator uses both image features and label information to detect fakes

### 3. Convolutional Conditional GAN
Replaces linear layers in the Generator with **transposed convolutions** and **bilinear upsampling**:

- Upsampling path: `(7×7) → (14×14) → (28×28)` using `ConvTranspose2d` or `Upsample + Conv2d`
- Discriminator remains linear (sufficient for MNIST complexity)

---

## Backpropagation & Training Details

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/GAN%20backpropagation%20step.png?raw=true" alt="GAN Backpropagation" width="700"/>

**Key training insights covered in the notebook:**

1. **Two Separate Optimizers** — Discriminator and Generator weights are updated independently
2. **Gradient Isolation** — `loss.backward()` computes gradients for both networks, but only the target network's optimizer calls `.step()`
3. **Gradient Zeroing** — Critical to zero gradients before each step to prevent accumulation across discriminator/generator updates
4. **Label Smoothing** — Soft labels (0.95 instead of 1.0) for real images to stabilize training

---

## Results

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/mnist%20results.png?raw=true" alt="MNIST Generation Results" width="500"/>

Generated MNIST digits after 200 epochs of training, showing clear improvement over VAE-based generation.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Latent Dimension | 100 |
| Batch Size | 64 |
| Epochs | 200 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |

---

## File Structure

```
Vanilla GAN/
├── Intro_to_Gans.ipynb    # Complete notebook with theory + 3 implementations
└── README.md              # This file
```

---

## References

- [Generative Adversarial Networks (Goodfellow et al., 2014)](../Papers/Vanilla%20GAN%20paper.pdf)
- [Conditional Generative Adversarial Nets (Mirza & Osindero, 2014)](../Papers/Conditional%20GAN%20(cGAN).pdf)
