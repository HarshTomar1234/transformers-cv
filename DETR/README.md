# DETR: Detection Transformer

**End-to-End Object Detection with Transformers**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Complete Architecture Overview](#complete-architecture-overview)
   - [Transformer Encoder](#transformer-encoder)
   - [Transformer Decoder](#transformer-decoder)
   - [Hungarian Matching and Loss](#hungarian-matching-and-loss)
3. [Resources](#resources)
4. [Inference Notebooks](#inference-notebooks)
5. [Implementation Status](#implementation-status)
6. [References](#references)

---

## Overview

DETR (Detection Transformer) is a groundbreaking approach to object detection that treats the problem as a direct set prediction task. Unlike traditional object detection methods that rely on hand-crafted components such as anchor boxes, non-maximum suppression (NMS), and region proposals, DETR uses a transformer encoder-decoder architecture to directly predict the final set of detections in a single forward pass.

### Key Innovations

- **End-to-End Detection**: Eliminates the need for many hand-designed components like NMS and anchor generation
- **Set-Based Global Loss**: Uses bipartite matching to force unique predictions via Hungarian algorithm
- **Transformer Architecture**: Leverages self-attention mechanisms to model global context and object relationships
- **Parallel Decoding**: Predicts all objects simultaneously using learned object queries

### Model Characteristics

| Aspect | Description |
|--------|-------------|
| Backbone | ResNet-50 or ResNet-101 for feature extraction |
| Encoder Layers | 6 transformer encoder layers with multi-head self-attention |
| Decoder Layers | 6 transformer decoder layers with self and cross-attention |
| Object Queries | 100 learned query embeddings (fixed set of predictions) |
| Output | Class labels + bounding box coordinates for each query |

---

## Architecture

This section contains detailed architectural diagrams created in Excalidraw to explain the inner workings of DETR. Each diagram provides comprehensive annotations explaining the data flow, mathematical operations, and design decisions.

---

### Complete Architecture Overview

The following diagram illustrates the complete DETR pipeline from input image to final predictions:

<p align="center">
<img src="DETR architecture overview.png" alt="DETR Complete Architecture" width="100%">
</p>

**Key Components Illustrated:**

1. **CNN Backbone (Part 1)**: ResNet processes the input image to extract spatial features, reducing the resolution while increasing channel depth

2. **Feature Map Processing (Part 2)**: The feature map is flattened and combined with positional encodings to preserve spatial information

3. **Transformer Encoder (Part 3)**: Processes the flattened feature map with self-attention to capture global relationships

4. **Transformer Decoder (Part 4)**: Takes learned object queries and attends to encoder output to produce object embeddings

5. **Prediction Heads (Part 5)**: FFN layers predict class probabilities and bounding box coordinates

6. **Hungarian Matching**: Bipartite matching algorithm assigns predictions to ground truth for loss computation

---

### Transformer Encoder

The encoder architecture with detailed attention mechanisms and positional encoding strategies:

<p align="center">
<img src="DETR Encoder structure.png" alt="DETR Encoder Structure" width="100%">
</p>

**Core Concepts Explained:**

**Positional Encoding Strategy:**
- DETR uses 2D sinusoidal positional encodings to maintain spatial relationships in the flattened feature map
- Positional encodings are added to Query (Q) and Key (K) vectors only, not to Values (V)
- This allows the attention weights to be spatially aware while keeping value content pure

**Encoder Block Operations:**
- Each encoder block consists of multi-head self-attention followed by a feed-forward network
- Layer normalization and residual connections are applied after each sub-layer
- The attention mechanism allows each spatial location to attend to all other locations globally

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where positional encodings (PE) are added: Q = features + PE, K = features + PE

---

### Transformer Decoder

The decoder architecture showing object queries, self-attention, and cross-attention mechanisms:

<p align="center">
<img src="DETR decoder structure.png" alt="DETR Decoder Structure" width="100%">
</p>

**Object Queries Explained:**

- Object queries are learnable embeddings (100 vectors of dimension d_model)
- Each query is responsible for detecting at most one object
- Queries develop specialization through training (e.g., certain queries learn to detect objects in specific image regions or of specific sizes)

**Decoder Attention Flow:**

1. **Self-Attention**: Object queries attend to each other to prevent duplicate detections and model inter-object relationships

2. **Cross-Attention**: Queries attend to encoder output to "look at" the image and gather object-specific information
   - Q comes from object queries
   - K und V come from encoder output (image features)

3. **Feed-Forward Network**: Transforms the attended features into object representations

**Why This Design Works:**
- Self-attention as a communication meeting: Queries coordinate to avoid detecting the same object twice
- Cross-attention as image probing: Each query asks "Where is my object?" and gathers relevant visual evidence
- The decoder iteratively refines object representations through multiple layers

---

### Hungarian Matching and Loss

The bipartite matching algorithm and loss computation explained step-by-step:

<p align="center">
<img src="hungarian matching and DETR loss.png" alt="Hungarian Matching and DETR Loss" width="100%">
</p>

**Hungarian Matching Process:**

1. **Cost Matrix Construction**: For N predictions and M ground truth objects, compute an NxM cost matrix where each entry represents the cost of matching prediction i to ground truth j

2. **Matching Cost Components:**
   - Classification cost: Negative log probability of the true class
   - Bounding box L1 cost: L1 distance between predicted and ground truth boxes
   - GIoU cost: Generalized Intersection over Union loss

3. **Optimal Assignment**: Apply the Hungarian algorithm to find the minimum cost bipartite matching

4. **One-to-One Matching**: Each ground truth is matched to exactly one prediction; unmatched predictions are assigned to "no object" class

**DETR Loss Function:**

```
L_total = L_class + lambda_L1 * L_bbox + lambda_giou * L_giou
```

Where:
- L_class: Cross-entropy loss for classification (with down-weighted "no object" class)
- L_bbox: L1 loss on normalized bounding box coordinates
- L_giou: Generalized IoU loss for scale-invariant box regression

**Why Hungarian Matching is Essential:**
- Ensures unique assignments: No two predictions are matched to the same ground truth
- Enables set prediction: The model learns to predict a fixed-size set where order does not matter
- Eliminates NMS: Post-processing suppression becomes unnecessary

---

## Resources

### Research Paper

The original DETR paper is included in this repository for reference:

| Resource | File | Description |
|----------|------|-------------|
| Original Paper | [DETR paper.pdf](DETR%20paper.pdf) | "End-to-End Object Detection with Transformers" by Carion et al. (ECCV 2020) |

### Sample Videos

The `videos/` folder contains sample inference outputs:

| File | Description |
|------|-------------|
| `funny fall.mp4` | Sample input video for inference testing |
| `output_annotated.mp4` | Annotated output video with DETR detections |

---

## Inference Notebooks

Two Jupyter notebooks are provided for running DETR inference using pre-trained models from Hugging Face:

### Image Inference

**Notebook**: [DETR_inference_on_image.ipynb](DETR%20inference%20notebooks/DETR_inference_on_image.ipynb)

This notebook demonstrates:
- Loading pre-trained DETR model from Hugging Face
- Running inference on single images
- Visualizing bounding box predictions with class labels
- Configuring confidence thresholds

**Quick Start:**
```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Run inference
image = Image.open("your_image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

### Video Inference

**Notebook**: [DETR_inference_on_video.ipynb](DETR%20inference%20notebooks/DETR_inference_on_video.ipynb)

This notebook demonstrates:
- Frame-by-frame DETR inference on video files
- Annotating video frames with detections
- Saving annotated output videos
- Handling various video formats

**Requirements:**
- Google Colab with GPU runtime (T4 recommended)
- transformers library
- OpenCV for video processing
- PIL for image handling

---

## Implementation Status

### Current State

| Component | Status |
|-----------|--------|
| Architecture Diagrams | Complete |
| Research Paper | Included |
| Pre-trained Model Inference | Complete (notebooks available) |
| Custom DETR Implementation | Planned |

### Planned Implementation

A complete from-scratch implementation of DETR is planned for this repository. The implementation will include:

**Backbone:**
- [ ] ResNet-50 feature extractor
- [ ] Feature map projection to transformer dimension

**Transformer Encoder:**
- [ ] Multi-head self-attention with 2D positional encoding
- [ ] Encoder layer with FFN and layer normalization
- [ ] Stack of 6 encoder layers

**Transformer Decoder:**
- [ ] Learnable object queries (100 queries)
- [ ] Self-attention among object queries
- [ ] Cross-attention with encoder output
- [ ] Stack of 6 decoder layers

**Prediction Heads:**
- [ ] Classification head (class + "no object")
- [ ] Bounding box regression head (center, width, height)

**Training Components:**
- [ ] Hungarian matching algorithm
- [ ] Set-based loss function (classification + bbox + GIoU)
- [ ] Training loop with COCO dataset

---

## References

### Original Paper

```bibtex
@inproceedings{carion2020end,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

### Key Resources

| Resource | Link |
|----------|------|
| Official Repository | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| Hugging Face Model | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) |
| Paper (arXiv) | [arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872) |

### Related Work

- **ViT (Vision Transformer)**: Foundation for image patch-based transformers
- **DINO**: Follow-up work with improved training stability
- **Deformable DETR**: Addresses slow convergence with deformable attention
- **DAB-DETR**: Uses anchor boxes as queries for better interpretability

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
