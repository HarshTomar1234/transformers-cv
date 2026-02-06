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
4. [Resources](#resources)
5. [Implementation Status](#implementation-status)
6. [References](#references)

---

## Overview

Swin Transformer (Shifted Window Transformer) is a hierarchical vision transformer that computes representations using shifted windows. Unlike ViT which processes the entire image as a single sequence, Swin Transformer restricts self-attention to non-overlapping local windows while enabling cross-window connections through a clever shifting mechanism.

This design achieves **linear computational complexity** with respect to image size, making it practical for high-resolution images and dense prediction tasks like object detection and semantic segmentation.

### Key Innovations

- **Hierarchical Feature Maps**: Produces multi-scale features like CNNs, enabling use as a backbone for various vision tasks
- **Shifted Windows**: Creates cross-window connections without the full cost of global attention
- **Linear Complexity**: O(n) computational cost instead of ViT's O(n²), enabling high-resolution image processing
- **Versatile Backbone**: State-of-the-art results on image classification, object detection, and semantic segmentation

### Model Characteristics

| Aspect | Description |
|--------|-------------|
| Patch Size | 4×4 pixels (smaller than ViT's 16×16 for finer granularity) |
| Window Size | 7×7 patches (default), attention computed within each window |
| Stages | 4 stages with progressively reduced spatial resolution |
| Feature Dimensions | [96, 192, 384, 768] across stages (Swin-T) |
| Attention Type | Window-based self-attention with shifted window mechanism |

---

## Architecture

This section contains detailed architectural diagrams explaining the inner workings of Swin Transformer. The diagrams provide comprehensive annotations explaining the data flow, mathematical operations, and design decisions.

---

### Complete Architecture Overview

The following diagram illustrates the complete Swin Transformer architecture:

<p align="center">
<img src="Swin Transformer architecture overview.png" alt="Swin Transformer Complete Architecture" width="100%">
</p>

**Key Components Illustrated:**

1. **Patch Partition + Linear Embedding**: Input image is divided into 4×4 patches and projected to embedding dimension

2. **Stage 1-4**: Each stage consists of Swin Transformer blocks with window attention, followed by patch merging (downsampling)

3. **Window Attention (W-MSA)**: Regular window-based multi-head self-attention within non-overlapping windows

4. **Shifted Window Attention (SW-MSA)**: Windows are shifted by half the window size to enable cross-window connections

5. **Patch Merging**: Reduces spatial resolution by 2× while doubling the channel dimension (similar to CNN pooling)

---

### Hierarchical Feature Maps

Unlike ViT which maintains the same resolution throughout, Swin Transformer produces hierarchical feature maps:

| Stage | Feature Map Size | Channels | Description |
|-------|------------------|----------|-------------|
| Stage 1 | H/4 × W/4 | 96 | Fine-grained features |
| Stage 2 | H/8 × W/8 | 192 | After first patch merging |
| Stage 3 | H/16 × W/16 | 384 | After second patch merging |
| Stage 4 | H/32 × W/32 | 768 | Coarse semantic features |

This hierarchical structure makes Swin Transformer suitable as a general-purpose backbone for dense prediction tasks, similar to how ResNet/FPN are used.

---

### Window-Based Self-Attention

The core efficiency of Swin Transformer comes from computing attention within local windows:

**Standard ViT Global Attention:**
```
Complexity: O((H×W)²) = O(n²)
For 224×224 image with 16×16 patches: 196 × 196 = 38,416 attention pairs
```

**Swin Window Attention:**
```
Complexity: O(M² × (H×W/M²)) = O(n×M²) = O(n)
M = window size (7×7 = 49 tokens per window)
Each window: 49 × 49 = 2,401 attention pairs
Total: Linear in image size
```

**The key insight**: Attention within 7×7 windows is sufficient for local feature learning. Cross-window connections are achieved through the shifted window mechanism.

---

## Key Concepts

### Shifted Windows

The fundamental problem with window-based attention is that windows are isolated — tokens at window boundaries cannot communicate with their neighbors in adjacent windows.

**Swin's Solution:** Alternate between two windowing configurations:

1. **Layer L (Regular Windows)**: Partition image into non-overlapping M×M windows starting from top-left (0, 0)

2. **Layer L+1 (Shifted Windows)**: Shift the window partition by (M/2, M/2) pixels

This shifting ensures that tokens separated by a boundary in layer L become part of the same window in layer L+1, enabling cross-window information flow.

### Cyclic Shift

When windows are shifted, border regions become irregularly sized, which is terrible for GPU efficiency. Swin uses **cyclic shift** to solve this:

1. **Problem**: After shifting, we get 9 sub-regions of varying sizes instead of 4 uniform windows
2. **Solution**: Roll the feature map using `torch.roll` — move top rows to bottom, left columns to right
3. **Result**: Back to exactly 4 windows of uniform M×M size, efficient for batched computation

This is like rolling a sheet of paper into a cylinder — content wraps around.

### Attention Masking

Cyclic shift creates a new problem: some windows now contain tokens from **spatially distant** regions of the original image (e.g., top-left corner next to bottom-right corner).

**The mask prevents false attention between artificially grouped tokens:**

| Token Pair Type | Mask Value | Effect |
|-----------------|------------|--------|
| Same original region (genuine neighbors) | 0 | Normal attention |
| Different regions (artificial neighbors from cyclic shift) | −∞ | Zero attention weight after softmax |

**Why −∞ works:**
```
softmax(logit + (-∞)) = e^(-∞) / Σ = 0 / Σ = 0
```

The masked positions contribute exactly zero to the output, while valid attention weights are automatically normalized to sum to 1.

> **📘 Deep Dive**: For a comprehensive 280+ line explanation of the masking mechanism, see [swin_transformer_masking_notes.md](swin_transformer_masking_notes.md)

---

## Resources

### Documentation

| Resource | File | Description |
|----------|------|-------------|
| Masking Notes | [swin_transformer_masking_notes.md](swin_transformer_masking_notes.md) | Comprehensive explanation of shifted windows, cyclic shift, and attention masking |
| Masking PDF | [swin_transformer_masking.pdf](swin_transformer_masking.pdf) | PDF version of the masking documentation |
| Interactive Demo | [swin_mask.html](swin_mask.html) | Interactive HTML visualization of the masking mechanism |

### Research Paper

| Resource | File | Description |
|----------|------|-------------|
| Original Paper | [Swin Transformer paper.pdf](Swin%20Transformer%20paper.pdf) | "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" by Liu et al. (ICCV 2021) |

### Editable Source

| Resource | File | Description |
|----------|------|-------------|
| Architecture Diagram | [Swin Transformer.excalidraw](Swin%20Transformer.excalidraw) | Excalidraw source file for the architecture diagram |

---

## Implementation Status

### Current State

| Component | Status |
|-----------|--------|
| Architecture Diagram | Complete |
| Masking Documentation | Complete (281 lines) |
| Interactive Demo | Complete |
| Research Paper | Included |
| From-Scratch Implementation | Planned |

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

- **ViT (Vision Transformer)**: Global attention on image patches — Swin's predecessor
- **Swin Transformer V2**: Scaling up to 3B parameters with improved training
- **CSwin Transformer**: Cross-shaped window attention for even better performance
- **Focal Transformer**: Combining local and global attention efficiently

### Comparison with ViT

| Aspect | ViT | Swin Transformer |
|--------|-----|------------------|
| Attention Scope | Global (all patches) | Local (within windows) |
| Complexity | O(n²) | O(n) |
| Feature Maps | Single scale | Hierarchical (multi-scale) |
| Resolution Support | Fixed (usually 224×224) | Flexible (any resolution) |
| Best For | Classification | Classification + Detection + Segmentation |

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
