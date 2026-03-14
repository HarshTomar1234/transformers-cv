# Transformers for Computer Vision

A comprehensive repository for understanding and implementing transformer-based architectures in computer vision. Contains detailed architecture explanations, annotated diagrams, inference notebooks, and from-scratch implementations.

---

## Architectures

| Architecture | Description | Status |
|--------------|-------------|--------|
| [ViT](./ViT/) | Vision Transformer -- Image classification with pure transformers | Complete (CIFAR-10 + ImageNet) |
| [DETR](./DETR/) | Detection Transformer -- End-to-end object detection | Documentation + Inference |
| [Swin Transformer](./Swin%20Transformer/) | Hierarchical Vision Transformer with shifted windows | Documentation + From-Scratch Implementation |
| [SAM](./SAM/) | Segment Anything Model -- Foundation model for image segmentation | Documentation + Inference (SAM + SAM 2) |
| [TimeSformer](./TimeSformer/) | Space-Time Attention for Video Understanding | Documentation + From-Scratch Implementation |
| [Flamingo VLM](./Flamingo%20VLM/) | Visual Language Model -- Few-shot multimodal learning | Documentation + Inference |
| [AutoEncoders](./AutoEncoders/) | Vanilla AE, VAE, VQ-VAE -- Unsupervised representation learning | Vanilla AE + VAE + VQ-VAE Complete (RVQ Planned) |
| [GAN](./GAN/) | Generative Adversarial Networks -- Vanilla GAN, Conditional GAN, DCGAN | Vanilla GAN + DCGAN Complete |
| [Diffusion](./Diffusion/) | Diffusion Models -- DDPM from scratch (FashionMNIST, CIFAR-10, CelebA) | DDPM Complete (3 Datasets) |
| [JEPA](./JEPA/) | Joint-Embedding Predictive Architecture -- I-JEPA, V-JEPA, V-JEPA 2, LLM-JEPA | Papers + Architecture Diagrams |
| [DeiT](./DeIT/) | Data-efficient Image Transformers | Paper Only |

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
│   ├── README.md                        # Detailed Swin documentation
│   ├── swin_transformer.py             # From-scratch implementation
│   ├── Swin_Transformer_coding.ipynb   # Step-by-step notebook + training
│   ├── Swin Transformer architecture overview.png
│   ├── swin_transformer_masking_notes.md
│   ├── swin_transformer_masking.pdf
│   ├── swin_mask.html                  # Interactive masking demo
│   ├── swin_region_ids.html            # Interactive region ID demo
│   ├── Swin Transformer.excalidraw
│   └── Swin Transformer paper.pdf
├── SAM/
│   ├── README.md                        # Detailed SAM documentation
│   ├── Segment_Anything_Model_(SAM).ipynb           # SAM inference notebook
│   ├── Segmentation_of_images_with_SAM_2.ipynb      # SAM 2 image segmentation
│   ├── Segment_video_with_SAM_2 .ipynb              # SAM 2 video segmentation
│   ├── result_videos/                   # SAM 2 video segmentation outputs
│   ├── Segment Anything Model (SAM).excalidraw
│   └── Segment Anything model paper.pdf
├── TimeSformer/
│   ├── README.md                        # Detailed TimeSformer documentation
│   ├── TimeSformer_implementation_from_scratch_V2.ipynb  # From-scratch implementation
│   ├── timesformer_code_accurate.html   # Interactive Pre-LN architecture diagram
│   ├── timeSfomer architecture.png      # Architecture diagram
│   ├── timeSformer architecture breakdown.excalidraw
│   ├── timeSformer paper.pdf            # Original research paper
│   ├── scripts/
│   │   ├── download_videos.py           # Download Kinetics-400 videos
│   │   └── extract_frames.py           # Extract frames from videos
│   └── data/                            # Downloaded dataset (gitignored)
├── Flamingo VLM/
│   ├── README.md                        # Detailed Flamingo documentation
│   ├── Flamingo_inference.ipynb         # Few-shot captioning with OpenFlamingo-9B
│   ├── Flamingo VLM  breakdown.png     # Architecture breakdown diagram
│   ├── Flamingo VLM architecture.excalidraw
│   └── Flamingo VLM paper.pdf          # Original research paper
├── AutoEncoders/
│   ├── README.md                        # AutoEncoders overview documentation
│   ├── images/                          # Theory diagrams (KL div, ELBO, VAE loss, VQ-VAE)
│   ├── Vanilla Autoencoder/             # From-scratch AE on MNIST & FashionMNIST
│   ├── VAE/                             # From-scratch VAE on FashionMNIST
│   └── VQVAE/                           # From-scratch VQ-VAE on FashionMNIST
├── GAN/
│   ├── README.md                        # GAN overview documentation
│   ├── Vanilla GAN/                     # Intro to GANs + 3 implementations (MNIST)
│   ├── DCGAN/                           # Deep Convolutional GAN (CelebA)
│   ├── images/                          # Architecture diagrams
│   ├── animated workflow/               # 10 interactive HTML architecture diagrams
│   └── Papers/                          # Original research papers
├── Diffusion/
│   └── DDPM/                            # Denoising Diffusion Probabilistic Models
│       ├── README.md                    # DDPM documentation with results
│       ├── DDPM_from_scratch.ipynb      # FashionMNIST 28×28
│       ├── DDPM_CIFAR10.ipynb           # CIFAR-10 32×32 RGB
│       ├── DDPM_CelebA.ipynb            # CelebA 64×64 RGB
│       ├── DDPM breakdown.excalidraw    # Architecture breakdown diagram
│       ├── DDPM paper.pdf               # Original research paper
│       └── images/                      # Architecture diagrams + training results
├── JEPA/
│   ├── README.md                        # JEPA family overview documentation
│   ├── architecture/                    # Architecture diagrams (I-JEPA, V-JEPA, V-JEPA 2, LLM-JEPA)
│   ├── I-JEPA paper.pdf                 # I-JEPA research paper
│   ├── V-JEPA paper.pdf                 # V-JEPA research paper
│   ├── V-JEPA 2 paper.pdf              # V-JEPA 2 research paper
│   └── llm-JEPA paper.pdf              # LLM-JEPA research paper
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
