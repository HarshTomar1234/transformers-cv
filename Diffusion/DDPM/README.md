# Denoising Diffusion Probabilistic Models (DDPM)

From-scratch implementations of **DDPM** on three datasets -- FashionMNIST, CIFAR-10, and CelebA -- using PyTorch.

**Paper:** [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

---

## Implementations

| Notebook | Dataset | Resolution | Channels | Training Time (T4) | Status |
|----------|---------|------------|----------|---------------------|--------|
| [DDPM_from_scratch.ipynb](./DDPM_from_scratch.ipynb) | FashionMNIST | 28x28 | 1 (grayscale) | ~45 min (50 epochs) | Verified |
| [DDPM_CIFAR10.ipynb](./DDPM_CIFAR10.ipynb) | CIFAR-10 | 32x32 | 3 (RGB) | ~5 hours (80 epochs) | Verified |
| [DDPM_CelebA.ipynb](./DDPM_CelebA.ipynb) | CelebA | 64x64 | 3 (RGB) | ~8-12 hours (100 epochs) | Requires Colab Pro |

---

## How DDPM Works

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/forward%20and%20backward%20DDPM.jpg?raw=true" alt="Forward and Backward DDPM" width="700"/>

### Forward Process -- Adding Noise

Given a clean image $x_0$, we progressively add Gaussian noise over $T$ timesteps:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) \, \mathbf{I})$$

Using the **reparameterization trick**, we can sample any noisy version directly:

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### Reverse Process -- Denoising / Generating

A U-Net neural network learns to predict the noise and undo each step:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} \, z$$

### Training Objective

Simple MSE loss between predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

---

## U-Net Architecture

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/U-Net%20model.jpg?raw=true" alt="U-Net Architecture" width="700"/>

The noise-predicting U-Net takes noisy image $x_t$ and timestep $t$ as input:

- **Sinusoidal Time Embeddings** -- Injected at every residual block via MLP projection
- **Encoder** -- Residual blocks with downsampling (AvgPool / strided conv)
- **Bottleneck** -- ResBlock, Self-Attention, ResBlock
- **Decoder** -- Residual blocks with upsampling + skip connections (concatenation)
- **GroupNorm(8)** normalization, **SiLU** activation throughout

### Model Configurations

| Configuration | FashionMNIST | CIFAR-10 | CelebA |
|--------------|-------------|----------|--------|
| Input channels | 1 | 3 | 3 |
| Image size | 28x28 | 32x32 | 64x64 |
| Base channels | 64 | 128 | 128 |
| Channel progression | 64 - 128 | 128 - 256 - 256 - 512 | 128 - 128 - 256 - 256 - 512 |
| Downsample levels | 2 (28 - 14 - 7) | 3 (32 - 16 - 8 - 4) | 4 (64 - 32 - 16 - 8 - 4) |
| Attention resolution | 7x7 | 8x8, 4x4 | 16x16, 8x8, 4x4 |

---

## Training Improvements

All three notebooks include these improvements over a basic DDPM:

| Feature | Details |
|---------|---------|
| **Learning Rate** | `2e-4` (AdamW optimizer) |
| **Gradient Clipping** | `max_norm = 1.0` -- prevents loss spikes |
| **EMA** | Exponential Moving Average (decay=0.9999) -- smoother, higher-quality samples |
| **GroupNorm(8)** | Proper group normalization (not instance norm) |
| **Noise Schedule** | Linear beta: 1e-4 to 0.02 (cosine option in CIFAR-10/CelebA) |

---

## Results

### Noise Schedule

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/noise_schedule.png?raw=true" alt="Noise Schedule" width="600"/>

The **linear beta schedule** gradually increases noise from beta_1 = 1e-4 to beta_T = 0.02, while the cumulative alpha_bar shows how signal degrades over timesteps.

---

### FashionMNIST (28x28 Grayscale)

**Training:** 50 epochs, loss converged from 0.083 to 0.036 with no spikes.

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/fashionmnist_loss.png?raw=true" alt="FashionMNIST Loss Curve" width="600"/>

**Forward Diffusion Process:**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/fashionmnist_forward_diffusion.png?raw=true" alt="Forward Diffusion" width="700"/>

**Generated Samples (EMA Model):**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/fashionmnist_samples.png?raw=true" alt="FashionMNIST Generated Samples" width="500"/>

The model generates recognizable FashionMNIST items -- sandals, shirts, bags, pants, sneakers -- with clear shapes and structures.

**EMA vs No EMA Comparison:**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/fashionmnist_ema_comparison.png?raw=true" alt="EMA vs No EMA" width="700"/>

> EMA (bottom row) produces noticeably smoother, cleaner samples compared to the raw model (top row).

---

### CIFAR-10 (32x32 RGB)

**Training:** 80 epochs, loss converged from 0.076 to 0.030, smooth and stable.

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/cifar10_loss.png?raw=true" alt="CIFAR-10 Loss Curve" width="600"/>

**Generated Samples (EMA Model):**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/cifar10_samples.png?raw=true" alt="CIFAR-10 Generated Samples" width="500"/>

Generated images show recognizable objects -- horses, cars, animals, people -- with realistic colors and compositions.

**Reverse Diffusion Process (Noise to Image):**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/cifar10_denoising.png?raw=true" alt="CIFAR-10 Denoising" width="700"/>

The reverse process shows clear coarse-to-fine generation: colored blobs form around t=400, structures emerge by t=300, and final details sharpen by t=100.

**64 Generated Samples:**

<img src="https://github.com/HarshTomar1234/transformers-cv/blob/main/Diffusion/DDPM/images/cifar10_64_samples.png?raw=true" alt="CIFAR-10 64 Samples" width="600"/>

---

### CelebA (64x64 RGB) -- Planned

The CelebA notebook is ready to train but requires Colab Pro for sufficient GPU time (~8-12 hours on T4).
Results will be added once training is complete.

---

## Repository Structure

```
DDPM/
├── DDPM_from_scratch.ipynb       # FashionMNIST (28x28) - Verified
├── DDPM_CIFAR10.ipynb            # CIFAR-10 (32x32) - Verified
├── DDPM_CelebA.ipynb             # CelebA (64x64) - Ready to train
├── DDPM paper.pdf                # Original DDPM paper
├── DDPM breakdown.excalidraw     # Architecture breakdown diagram
├── README.md                     # This file
└── images/
    ├── U-Net model.jpg           # U-Net architecture diagram
    ├── forward and backward DDPM.jpg
    ├── noise_schedule.png        # Beta and alpha_bar schedule plots
    ├── fashionmnist_loss.png     # FashionMNIST training loss curve
    ├── fashionmnist_forward_diffusion.png
    ├── fashionmnist_samples.png  # Generated FashionMNIST samples
    ├── fashionmnist_ema_comparison.png
    ├── cifar10_loss.png          # CIFAR-10 training loss curve
    ├── cifar10_samples.png       # Generated CIFAR-10 samples
    ├── cifar10_denoising.png     # Reverse diffusion visualization
    └── cifar10_64_samples.png    # 64 generated samples grid
```

---

## Key Takeaways

1. **DDPM training is stable** -- With gradient clipping and proper LR, no loss spikes
2. **EMA significantly improves sample quality** -- Visibly smoother outputs
3. **Architecture scales naturally** -- Same building blocks (ResBlock, Attention, Time Embedding) work across resolutions
4. **From-scratch implementation** validates the paper's core ideas -- simple noise prediction loss produces impressive generative results

---

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
