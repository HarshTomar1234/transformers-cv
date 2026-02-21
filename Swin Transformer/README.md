# Swin Transformer: Hierarchical Vision Transformer

**Hierarchical Vision Transformer using Shifted Windows**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Complete Architecture Overview](#complete-architecture-overview)
   - [Hierarchical Feature Maps](#hierarchical-feature-maps)
   - [Window-Based Self-Attention](#window-based-self-attention)
3. [Key Concepts](#key-concepts)
   - [Shifted Windows](#shifted-windows)
   - [Cyclic Shift](#cyclic-shift)
   - [Attention Masking](#attention-masking)
4. [From-Scratch Implementation](#from-scratch-implementation)
5. [Resources](#resources)
6. [References](#references)

---

## Overview

Swin Transformer (Shifted Window Transformer) is a hierarchical vision transformer that computes representations using shifted windows. Unlike ViT which processes the entire image as a single sequence, Swin Transformer restricts self-attention to non-overlapping local windows while enabling cross-window connections through a shifting mechanism.

This design achieves **linear computational complexity** with respect to image size, making it practical for high-resolution images and dense prediction tasks like object detection and semantic segmentation.

### Key Innovations

- **Hierarchical Feature Maps**: Produces multi-scale features like CNNs, enabling use as a backbone for various vision tasks
- **Shifted Windows**: Creates cross-window connections without the full cost of global attention
- **Linear Complexity**: O(n) computational cost instead of ViT's O(n^2), enabling high-resolution image processing
- **Versatile Backbone**: State-of-the-art results on image classification, object detection, and semantic segmentation

### Model Characteristics

| Aspect | Description |
|--------|-------------|
| Patch Size | 4x4 pixels (smaller than ViT's 16x16 for finer granularity) |
| Window Size | 7x7 patches (default), attention computed within each window |
| Stages | 4 stages with progressively reduced spatial resolution |
| Feature Dimensions | [96, 192, 384, 768] across stages (Swin-T) |
| Attention Type | Window-based self-attention with shifted window mechanism |

---

## Architecture

### Complete Architecture Overview

The following diagram illustrates the complete Swin Transformer architecture:

<p align="center">
<img src="Swin Transformer architecture overview.png" alt="Swin Transformer Complete Architecture" width="100%">
</p>

**Key Components Illustrated:**

1. **Patch Partition + Linear Embedding**: Input image is divided into 4x4 patches and projected to embedding dimension

2. **Stage 1-4**: Each stage consists of Swin Transformer blocks with window attention, followed by patch merging (downsampling)

3. **Window Attention (W-MSA)**: Regular window-based multi-head self-attention within non-overlapping windows

4. **Shifted Window Attention (SW-MSA)**: Windows are shifted by half the window size to enable cross-window connections

5. **Patch Merging**: Reduces spatial resolution by 2x while doubling the channel dimension (similar to CNN pooling)

---

### Hierarchical Feature Maps

Unlike ViT which maintains the same resolution throughout, Swin Transformer produces hierarchical feature maps:

| Stage | Feature Map Size | Channels | Description |
|-------|------------------|----------|-------------|
| Stage 1 | H/4 x W/4 | 96 | Fine-grained features |
| Stage 2 | H/8 x W/8 | 192 | After first patch merging |
| Stage 3 | H/16 x W/16 | 384 | After second patch merging |
| Stage 4 | H/32 x W/32 | 768 | Coarse semantic features |

This hierarchical structure makes Swin Transformer suitable as a general-purpose backbone for dense prediction tasks, similar to how ResNet/FPN are used.

---

### Window-Based Self-Attention

The core efficiency of Swin Transformer comes from computing attention within local windows:

**Standard ViT Global Attention:**
```
Complexity: O((H*W)^2) = O(n^2)
For 224x224 image with 16x16 patches: 196 x 196 = 38,416 attention pairs
```

**Swin Window Attention:**
```
Complexity: O(M^2 * (H*W / M^2)) = O(n * M^2) = O(n)
M = window size (7x7 = 49 tokens per window)
Each window: 49 x 49 = 2,401 attention pairs
Total: Linear in image size
```

The key insight: attention within 7x7 windows is sufficient for local feature learning. Cross-window connections are achieved through the shifted window mechanism.

---

## Key Concepts

### Shifted Windows

The fundamental problem with window-based attention is that windows are isolated -- tokens at window boundaries cannot communicate with their neighbors in adjacent windows.

**Swin's Solution:** Alternate between two windowing configurations:

1. **Layer L (Regular Windows)**: Partition image into non-overlapping MxM windows starting from top-left (0, 0)

2. **Layer L+1 (Shifted Windows)**: Shift the window partition by (M/2, M/2) pixels

This shifting ensures that tokens separated by a boundary in layer L become part of the same window in layer L+1, enabling cross-window information flow.

### Cyclic Shift

When windows are shifted, border regions become irregularly sized, which is problematic for GPU efficiency. Swin uses **cyclic shift** to solve this:

1. **Problem**: After shifting, we get 9 sub-regions of varying sizes instead of 4 uniform windows
2. **Solution**: Roll the feature map using `torch.roll` -- move top rows to bottom, left columns to right
3. **Result**: Back to exactly 4 windows of uniform MxM size, efficient for batched computation

### Attention Masking

Cyclic shift creates a new problem: some windows now contain tokens from **spatially distant** regions of the original image (e.g., top-left corner next to bottom-right corner).

**The mask prevents false attention between artificially grouped tokens:**

| Token Pair Type | Mask Value | Effect |
|-----------------|------------|--------|
| Same original region (genuine neighbors) | 0 | Normal attention |
| Different regions (artificial neighbors from cyclic shift) | -inf | Zero attention weight after softmax |

**Why -inf works:**
```
softmax(logit + (-inf)) = e^(-inf) / sum = 0 / sum = 0
```

The masked positions contribute exactly zero to the output, while valid attention weights are automatically normalized to sum to 1.

> **Deep Dive**: For a comprehensive 280+ line explanation of the masking mechanism, see [swin_transformer_masking_notes.md](swin_transformer_masking_notes.md)

---

## From-Scratch Implementation

A complete 4-stage Swin Transformer, built from scratch in PyTorch.

### Components

| Component | Description |
|-----------|-------------|
| `window_partition` / `window_reverse` | Partition feature maps into local windows and merge back |
| `WindowAttention` | Multi-head self-attention with relative position bias |
| `SwinBlock` | W-MSA / SW-MSA + MLP with pre-norm residual connections |
| `PatchMerging` | Spatial downsampling: 2x2 concat (4C) then project to 2C |
| `PatchEmbed` | Strided convolution to tokenize raw pixels |
| `SwinStage` | Stack of alternating W-MSA and SW-MSA blocks |
| `SwinTransformer` | Full 4-stage model with classification head |

### Files

| File | Description |
|------|-------------|
| [swin_transformer.py](swin_transformer.py) | Standalone module -- all components in one importable file |
| [Swin_Transformer_coding.ipynb](Swin_Transformer_coding.ipynb) | Step-by-step notebook with inline implementation and training |

### Configuration (32x32 input)

```
Stage 1:  16x16,   64ch,  win=4  |  2 blocks
Stage 2:   8x8,   128ch,  win=4  |  2 blocks
Stage 3:   4x4,   256ch,  win=4  |  6 blocks
Stage 4:   2x2,   512ch,  win=2  |  2 blocks (no shift)
```

Parameters: ~12.2M

### Bug Fixes Applied (vs. Original Notebook)

During the from-scratch implementation, the following bugs were identified and fixed:

| Issue | What Was Wrong | Fix |
|-------|----------------|-----|
| Swapped arguments | `WindowAttention(dim, win_size, heads)` -- num_heads and win_size swapped | Correct argument order |
| No multi-head reshape | Q, K, V stayed as `(B_, N, C)`, never split into heads | Fused QKV + reshape to `(B_, h, N, d)` |
| Returns None | softmax + output was inside `if mask is not None` block | Moved outside the conditional |
| Float division | `window_reverse` used `/` for batch size (produces float) | Integer division `//` |
| Dim mismatch | `LayerNorm(merged_dim)` but only stage 1 ran (different dim) | Full 4-stage model with correct dims |
| Unused import | `from os import pread` | Removed |

---

## Resources

### Documentation

| Resource | File | Description |
|----------|------|-------------|
| Masking Notes | [swin_transformer_masking_notes.md](swin_transformer_masking_notes.md) | Comprehensive explanation of shifted windows, cyclic shift, and attention masking |
| Masking PDF | [swin_transformer_masking.pdf](swin_transformer_masking.pdf) | PDF version of the masking documentation |
| Interactive Demo | [swin_mask.html](swin_mask.html) | Interactive HTML visualization of the masking mechanism |
| Region ID Demo | [swin_region_ids.html](swin_region_ids.html) | Interactive HTML visualization of region IDs |

### Research Paper

| Resource | File | Description |
|----------|------|-------------|
| Original Paper | [Swin Transformer paper.pdf](Swin%20Transformer%20paper.pdf) | "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" by Liu et al. (ICCV 2021) |

### Editable Source

| Resource | File | Description |
|----------|------|-------------|
| Architecture Diagram | [Swin Transformer.excalidraw](Swin%20Transformer.excalidraw) | Excalidraw source file for the architecture diagram |

---

## References

### Original Paper

```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

### Key Resources

| Resource | Link |
|----------|------|
| Official Repository | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) |
| Hugging Face Model | [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) |
| Paper (arXiv) | [arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030) |

### Related Work

- **ViT (Vision Transformer)**: Global attention on image patches -- Swin's predecessor
- **Swin Transformer V2**: Scaling up to 3B parameters with improved training
- **CSwin Transformer**: Cross-shaped window attention for further performance gains
- **Focal Transformer**: Combining local and global attention efficiently

### Comparison with ViT

| Aspect | ViT | Swin Transformer |
|--------|-----|------------------|
| Attention Scope | Global (all patches) | Local (within windows) |
| Complexity | O(n^2) | O(n) |
| Feature Maps | Single scale | Hierarchical (multi-scale) |
| Resolution Support | Fixed (usually 224x224) | Flexible (any resolution) |
| Best For | Classification | Classification + Detection + Segmentation |

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
