# Generative Adversarial Networks (GANs)

A comprehensive exploration of GAN architectures — from the mathematical foundations to fully convolutional models generating realistic face images.

---

## Implementations

| Architecture | Dataset | Description |
|-------------|---------|-------------|
| [Vanilla GAN](./Vanilla%20GAN/) | MNIST | Unconditional, Conditional, and Convolutional GANs with full math derivation | 
| [DCGAN](./DCGAN/) | CelebA | Deep Convolutional GAN generating 64×64 face images | 

---

## GAN Architecture Overview

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/GAN/images/GAN%20architecture.png?raw=true" alt="GAN Architecture" width="700"/>

GANs consist of two competing neural networks:
- **Generator** — Transforms random noise into realistic images
- **Discriminator** — Distinguishes between real and generated images

The adversarial training process is captured by the minimax objective:

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{data}} [\log D(x)] + \mathbb{E}_{z\sim p_z} [\log(1 - D(G(z)))]$$

---

## What's Covered

### Mathematical Foundations (in Vanilla GAN)
- Complete derivation from BCE Loss to the GAN objective
- Optimal discriminator proof: $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$
- Connection to **Jensen-Shannon Divergence**
- Backpropagation mechanics with dual optimizers

### Progressive Complexity
1. **Linear Unconditional GAN** — Fully-connected generator/discriminator on MNIST
2. **Conditional GAN** — Class-conditioned generation using learned embeddings
3. **Convolutional GAN** — Upsampling with transposed convolutions
4. **DCGAN** — Full convolutional architecture on CelebA faces (64×64 RGB)

---

## Interactive Architecture Diagrams

The `animated workflow/` directory contains interactive HTML visualizations for various GAN architectures:

| File | Architecture |
|------|-------------|
| `gan_architecture.html` | Vanilla GAN |
| `dcgan_architecture.html` | DCGAN |
| `conditional_gan_architecture.html` | Conditional GAN |
| `wgan_architecture.html` | Wasserstein GAN |
| `stylegan_architecture.html` | StyleGAN |
| `progressive_gan_architecture.html` | Progressive GAN |
| `srgan_architecture.html` | Super Resolution GAN |
| `pix2pix_architecture.html` | Pix2Pix |
| `cyclegan_architecture.html` | CycleGAN |
| `infogan_architecture.html` | InfoGAN |

---

## Repository Structure

```
GAN/
├── Vanilla GAN/
│   ├── Intro_to_Gans.ipynb                   # Theory + 3 GAN implementations
│   └── README.md                              # Vanilla GAN documentation
├── DCGAN/
│   ├── deep_convolutional_gan(DCGAN).ipynb    # DCGAN on CelebA
│   └── README.md                              # DCGAN documentation
├── images/
│   ├── GAN architecture.png                   # GAN architecture diagram
│   ├── GAN backpropagation step.png           # Backprop visualization
│   ├── Conditional GAN.png                    # cGAN architecture
│   ├── DCGAN architecture.png                 # DCGAN architecture
│   ├── DCGAN.png                              # DCGAN generator detail
│   └── mnist results.png                      # Generated MNIST samples
├── animated workflow/                          # 10 interactive HTML architecture diagrams
├── Papers/
│   ├── Vanilla GAN paper.pdf
│   ├── DCGAN paper.pdf
│   └── Conditional GAN (cGAN).pdf
└── README.md                                  # This file
```

---

## References

- [Generative Adversarial Networks (Goodfellow et al., 2014)](./Papers/Vanilla%20GAN%20paper.pdf)
- [Unsupervised Representation Learning with DCGANs (Radford et al., 2016)](./Papers/DCGAN%20paper.pdf)
- [Conditional Generative Adversarial Nets (Mirza & Osindero, 2014)](./Papers/Conditional%20GAN%20(cGAN).pdf)
