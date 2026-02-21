# SAM: Segment Anything Model

**A Foundation Model for Image Segmentation**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Image Encoder](#image-encoder)
   - [Prompt Encoder](#prompt-encoder)
   - [Mask Decoder](#mask-decoder)
3. [Inference Notebook](#inference-notebook)
4. [Resources](#resources)
5. [Implementation Status](#implementation-status)
6. [References](#references)

---

## Overview

SAM (Segment Anything Model) is a foundation model for image segmentation developed by Meta AI. It was trained on the SA-1B dataset containing over 1 billion masks from 11 million images, making it capable of segmenting any object in any image given appropriate prompting.

The model introduces a **promptable segmentation** paradigm: given an input image and a prompt (point, box, or text), SAM produces a valid segmentation mask. This design allows zero-shot transfer to new image domains and tasks without additional training.

### Key Innovations

- **Promptable Segmentation**: Accepts diverse prompt types (points, bounding boxes, masks, text) to specify what to segment
- **Foundation Model Approach**: Pre-trained on SA-1B at massive scale, generalizes to unseen domains zero-shot
- **Efficient Architecture**: Separates the heavy image encoder (runs once) from the lightweight prompt encoder and mask decoder (run per prompt), enabling real-time interactive use
- **Ambiguity-Aware**: Outputs multiple plausible masks for ambiguous prompts, ranked by confidence

### Model Variants

| Variant | Image Encoder | Parameters | Description |
|---------|---------------|------------|-------------|
| SAM ViT-H | ViT-Huge | 636M | Highest quality, largest model |
| SAM ViT-L | ViT-Large | 308M | Balanced quality and speed |
| SAM ViT-B | ViT-Base | 91M | Fastest, suitable for interactive use |

---

## Architecture

SAM consists of three components that work together:

```
Input Image              Prompt (point/box/mask)
    |                          |
    v                          v
Image Encoder            Prompt Encoder
(ViT-H/L/B)           (sparse + dense)
    |                          |
    v                          v
Image Embedding -----> Mask Decoder <----- Prompt Tokens
                          |
                          v
                   Predicted Masks + IoU Scores
```

---

### Image Encoder

The heaviest component. A Vision Transformer (ViT) pre-trained with MAE (Masked Autoencoder) processes the input image into a dense embedding.

| Aspect | Detail |
|--------|--------|
| Input | 1024x1024 image (resized and padded) |
| Architecture | ViT with 14x14 windowed attention + 4 global attention blocks |
| Output | 64x64x256 image embedding |
| Key Property | Runs only once per image, amortized across multiple prompts |

The image encoder is the computational bottleneck. By decoupling it from the prompt-dependent components, SAM enables real-time interactive segmentation: encode the image once, then decode masks for each new prompt in milliseconds.

---

### Prompt Encoder

Converts user prompts into token embeddings that the mask decoder can consume.

**Sparse prompts** (points, boxes, text):

| Prompt Type | Encoding |
|-------------|----------|
| Points | Positional encoding + learned foreground/background embedding |
| Boxes | Positional encoding of two corner points |
| Text | CLIP text encoder (when available) |

**Dense prompts** (masks):
- Downscaled using convolutions and added element-wise to the image embedding

---

### Mask Decoder

A lightweight transformer decoder that combines image and prompt information to produce segmentation masks.

| Aspect | Detail |
|--------|--------|
| Design | Modified transformer decoder with 2 layers |
| Inputs | Image embedding + prompt tokens + output tokens |
| Attention | Bidirectional cross-attention (prompt-to-image and image-to-prompt) |
| Outputs | 3 masks (for ambiguity) + IoU confidence score for each |
| Upsampling | Transposed convolution to go from 64x64 to 256x256, then bilinear to full resolution |

The decoder is deliberately lightweight so that multiple masks can be predicted efficiently for different prompts on the same image.

---

## Inference Notebook

**Notebook**: [Segment_Anything_Model_(SAM).ipynb](Segment_Anything_Model_(SAM).ipynb)

This notebook demonstrates SAM inference using the official `segment-anything` library with the `sam_vit_b` checkpoint.

**What the notebook covers:**
- Installing dependencies and downloading the SAM ViT-B checkpoint
- Loading an image and initializing the SAM predictor
- Running inference with **point prompts** (foreground/background points)
- Running inference with **box prompts** (bounding box coordinates)
- Visualizing segmentation masks with confidence scores
- Handling multi-mask output for ambiguous prompts

**Usage pattern:**
```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device="cuda")

predictor = SamPredictor(sam)
predictor.set_image(image)

# Point prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),   # 1 = foreground
    multimask_output=True,
)
```

**Requirements:**
- Google Colab with GPU runtime (T4 recommended)
- `segment-anything` (installed from GitHub)
- PyTorch, OpenCV, matplotlib

---

## Resources

### Research Paper

| Resource | File | Description |
|----------|------|-------------|
| Original Paper | [Segment Anything model paper.pdf](Segment%20Anything%20model%20paper.pdf) | "Segment Anything" by Kirillov et al. (ICCV 2023) |

### Architecture Diagram

| Resource | File | Description |
|----------|------|-------------|
| Editable Source | [Segment Anything Model (SAM).excalidraw](Segment%20Anything%20Model%20(SAM).excalidraw) | Excalidraw source file for the architecture diagram |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Architecture Diagram | Complete |
| Research Paper | Included |
| Pre-trained Model Inference | Complete (notebook available) |
| From-Scratch Implementation | Planned |

---

## References

### Original Paper

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and Dollar, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4015--4026},
  year={2023}
}
```

### Key Resources

| Resource | Link |
|----------|------|
| Official Repository | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| Paper (arXiv) | [arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643) |
| SA-1B Dataset | [ai.meta.com/datasets/segment-anything](https://ai.meta.com/datasets/segment-anything/) |

### Related Work

- **SAM 2**: Extension to video segmentation with memory mechanism
- **FastSAM**: Real-time segment anything using YOLOv8-based architecture
- **MobileSAM**: Lightweight SAM with distilled ViT-Tiny image encoder
- **EfficientSAM**: Leveraging masked image pre-training for efficient SAM

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
