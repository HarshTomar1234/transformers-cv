# Transformers for Computer Vision

A comprehensive repository for understanding and implementing transformer-based architectures in computer vision. This repository contains detailed architecture explanations, annotated diagrams, inference notebooks, and from-scratch implementations.

---

## Architectures

| Architecture | Description | Status |
|--------------|-------------|--------|
| [ViT](./ViT/) | Vision Transformer - Image classification with pure transformers | Complete (CIFAR-10 + ImageNet) |
| [DETR](./DETR/) | Detection Transformer - End-to-end object detection | Documentation Complete |
| [DeiT](./DeIT/) | Data-efficient Image Transformers | Paper Only |
| [Swin Transformer](./Swin%20Transformer/) | Hierarchical Vision Transformer with shifted windows | Documentation Complete |

---

## Repository Structure

```
transformers-CV/
├── ViT/
│   ├── README.md                        # Detailed ViT documentation
│   ├── imgs/                            # Architecture diagrams
│   ├── VisionTransformer.ipynb          # Main notebook
│   ├── Vision Transformer on CIFAR-10/  # From-scratch implementation
│   ├── Vision transformer on ImageNet/  # ImageNet training code
│   └── Vision Transformer paper.pdf     # Original research paper
├── DETR/
│   ├── README.md                        # Detailed DETR documentation
│   ├── architecture_diagrams/           # Excalidraw architecture diagrams
│   ├── DETR inference notebooks/        # Image and video inference
│   ├── DETR paper.pdf                   # Original research paper
│   └── videos/                          # Sample inference outputs
├── Swin Transformer/
│   ├── Swin Transformer architecture overview.png  # Architecture diagram
│   ├── swin_transformer_masking_notes.md           # Detailed masking explanation
│   ├── swin_transformer_masking.pdf                # Masking documentation PDF
│   ├── swin_mask.html                              # Interactive masking demo
│   ├── Swin Transformer.excalidraw                 # Editable diagram source
│   └── Swin Transformer paper.pdf                  # Original research paper
├── DeIT/
│   └── DeIT paper.pdf                   # DeiT research paper
├── .gitignore
├── LICENSE
└── README.md
```

---

## Goals

- Deep understanding of transformer architectures in CV through detailed diagrams
- Practical inference notebooks using pre-trained models
- From-scratch implementations for learning purposes
- Comprehensive documentation with mathematical explanations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
