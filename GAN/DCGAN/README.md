# Deep Convolutional GAN (DCGAN)

A from-scratch PyTorch implementation of DCGAN trained on the **CelebA** faces dataset, generating realistic 64×64 face images using a fully convolutional architecture.

---

## Architecture

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/DCGAN%20architecture.png?raw=true" alt="DCGAN Architecture" width="700"/>

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/DCGAN.png?raw=true" alt="DCGAN Generator" width="700"/>

DCGAN extends the Vanilla GAN by replacing all fully-connected layers with **deep convolutional layers**, enabling the model to learn spatial hierarchies for higher-quality image generation.

### Generator
The Generator takes a noise vector `(B × 100 × 1 × 1)` and progressively upsamples it through transposed convolutions:

```
(B × 100 × 1 × 1)  →  ConvTranspose2d  →  (B × 1024 × 4 × 4)
(B × 1024 × 4 × 4) →  ConvTranspose2d  →  (B × 512 × 8 × 8)
(B × 512 × 8 × 8)  →  ConvTranspose2d  →  (B × 256 × 16 × 16)
(B × 256 × 16 × 16)→  ConvTranspose2d  →  (B × 128 × 32 × 32)
(B × 128 × 32 × 32)→  ConvTranspose2d  →  (B × 3 × 64 × 64)
```

### Discriminator
The Discriminator mirrors the Generator — using strided convolutions to downsample the input image to a single real/fake prediction.

---

## Key Implementation Details from the Paper

| Detail | Description |
|--------|-------------|
| **BatchNorm** | Applied to every layer except Generator output and Discriminator input |
| **Generator Activation** | ReLU in all layers, Tanh on the output |
| **Discriminator Activation** | LeakyReLU (allows negative gradients to flow for misclassified images) |
| **No FC Layers** | Fully convolutional — no fully-connected layers anywhere |
| **Weight Initialization** | All Conv2d and BatchNorm weights initialized from $\mathcal{N}(0, 0.02)$ |

---

## Optimizer Configuration

The notebook provides a detailed explanation of **Adam's beta parameters** and why they matter for GAN training:

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Learning Rate** | 1e-4 | — |
| **β₁** | 0.5 (default: 0.9) | Reduced to make the optimizer more responsive to rapid gradient changes between generator and discriminator |
| **β₂** | 0.999 (default) | Kept high for smoother gradients and stable training despite adversarial dynamics |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CelebA Faces |
| Image Size | 64 × 64 |
| Channels | 3 (RGB) |
| Latent Dimension | 100 |
| Batch Size | 64 |
| Epochs | 10 |
| Loss Function | BCEWithLogitsLoss |
| Label Smoothing | 0.05 |

---

## File Structure

```
DCGAN/
├── deep_convolutional_gan(DCGAN).ipynb   # Full implementation + training
└── README.md                             # This file
```

---

## References

- [Unsupervised Representation Learning with Deep Convolutional GANs (Radford et al., 2016)](../Papers/DCGAN%20paper.pdf)
- [Intro to GANs (Vanilla GAN notebook)](../Vanilla%20GAN/Intro_to_Gans.ipynb) — Mathematical foundations
