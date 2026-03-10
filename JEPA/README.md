# JEPA (Joint-Embedding Predictive Architecture)

A collection of JEPA-family architectures — a paradigm for self-supervised learning proposed by Yann LeCun that predicts in **representation space** rather than pixel/token space. This avoids the fundamental problems of generative approaches (wasting capacity on irrelevant pixel-level details) and contrastive methods (reliance on hand-crafted augmentations and negative pairs).

> **Note:** This folder currently contains research papers and architecture diagrams. Code implementations may be added in the future as the papers are studied in depth.

---

## Why JEPA? — The Core Problem

Traditional self-supervised learning falls into two camps, both with significant limitations:

**Generative Methods** (MAE, autoencoders, diffusion models) reconstruct the input at the pixel level. This forces the model to waste capacity predicting every high-frequency detail — texture noise, exact color values, lighting artifacts — none of which carry semantic meaning. The model gets bogged down in "what does this pixel look like?" instead of "what is happening in this scene?"

**Contrastive / Invariance-Based Methods** (SimCLR, DINO, MoCo) rely on hand-crafted augmentations (crops, color jitter, flips) to create positive pairs and push apart negative pairs. The quality of learned representations is directly tied to the quality of augmentation design — a strong inductive bias that may not transfer across domains.

**JEPA's insight:** Instead of predicting pixels or relying on augmentations, predict the *representation* of a target region from the representation of a context region. All prediction happens in a learned latent space. This forces the encoder to capture high-level semantic structure, because that is what is predictable in embedding space — low-level noise is inherently unpredictable and gets filtered out.

---

## The JEPA Family

### 1. I-JEPA — Image-based Joint-Embedding Predictive Architecture

*Assran et al., 2023 — "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"*

![I-JEPA Architecture](./architecture/I-JEPA.png)

**Architecture Breakdown:**

I-JEPA operates on a single image. The image is divided into patches, and a multi-block masking strategy selects context blocks (visible) and target blocks (masked). The three core components are:

- **Context Encoder (x-encoder)** — A Vision Transformer that processes *only* the visible context patches, producing patch-level representations. By restricting input to visible patches only, the encoder never sees the targets directly.

- **Target Encoder (y-encoder)** — Processes the *full* unmasked image to produce target representations. Updated via Exponential Moving Average (EMA) of the context encoder parameters — no gradients flow through it. This provides stable, slowly evolving targets that prevent representation collapse.

- **Predictor** — A lightweight transformer that takes context encoder outputs along with positional mask tokens (indicating *where* the targets are) and predicts the target representations. The loss is L2 distance between predicted and actual target embeddings.

**Key design choice — Multi-block masking:** Target blocks must be large enough to require semantic understanding (not just local texture interpolation), and the context block must be spatially distributed enough to provide meaningful long-range information. This is what separates I-JEPA from MAE — MAE predicts pixels of small random patches, I-JEPA predicts representations of large semantic regions.

**Results:** Competitive with DINO and MAE on ImageNet linear probing, while being significantly more compute-efficient. Notably outperforms pixel-reconstruction methods on tasks requiring semantic understanding (object counting, depth prediction).

---

### 2. V-JEPA — Video Joint-Embedding Predictive Architecture

*Bardes et al., 2024 — "V-JEPA: Latent Video Prediction for Visual Representation Learning"*

![V-JEPA Architecture](./architecture/V-JEPA.png)

**Architecture Breakdown:**

V-JEPA extends the JEPA paradigm to video by masking spatio-temporal regions (not just spatial patches). The model must predict representations of missing video regions — which requires understanding motion, object permanence, and temporal dynamics.

- **Masking Strategy** — Uses multi-block spatio-temporal masking with very high masking ratios (around 90%). Video is divided into 3D "tubelets" (spatial patch x temporal extent), and large contiguous blocks are masked out. The extreme masking ratio forces the model to reason about scene dynamics rather than relying on spatial interpolation.

- **Context Encoder** — A Video Vision Transformer (ViViT-style) that processes only the visible spatio-temporal tokens.

- **Predictor** — Takes visible token representations and positional mask tokens, predicts embeddings of masked spatio-temporal regions.

- **Target Encoder** — EMA of the context encoder, processes the full unmasked video to provide stable prediction targets.

**What makes V-JEPA different from video MAE:** Video MAE reconstructs pixel-level frames — this is extremely expensive for video (high dimensionality) and wastes capacity on temporally redundant pixel details. V-JEPA predicts in latent space, focusing on what *changes meaningfully* across time rather than what every pixel looks like frame-by-frame.

**Results:** Learns representations that capture motion understanding and temporal reasoning *without* any pre-trained image encoder, text supervision, negative examples, or human annotations. Strong performance on action recognition benchmarks with frozen backbone evaluation.

---

### 3. V-JEPA 2 — Scaling to World Models

*Assran et al., 2025 — "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"*

![V-JEPA 2 Architecture](./architecture/V-JEPA-2.png)

**Architecture Breakdown:**

V-JEPA 2 scales the V-JEPA framework to internet-scale video data (over 1 million hours) and extends its use to robotic planning and world modeling. The key insight is that a video encoder trained with JEPA objectives can serve as a "learned simulator" of physical dynamics — enabling embodied agents to plan actions without task-specific training.

**Scaling and Training:**
- Trained on massive web-scale video datasets, combined with a relatively small amount of real-world robot interaction data (~62 hours)
- Despite the small robotics data, the model generalizes effectively because the web-scale self-supervised pretraining provides a strong prior about how the physical world behaves

**Applications beyond representation learning:**
- **Video Understanding** — State-of-the-art performance on motion understanding benchmarks (Something-Something v2, Epic-Kitchens-100), action anticipation, and video question-answering
- **Simulation** — The learned latent space functions as a world model, simulating how scenes evolve over time
- **Zero-shot Robotic Planning** — The model can plan pick-and-place actions in novel environments without any task-specific fine-tuning, using the learned world model to predict outcomes of actions in latent space

**Why this matters:** V-JEPA 2 demonstrates that JEPA-style objectives, when scaled, naturally produce representations useful for planning and control — not just classification. This aligns with Yann LeCun's broader vision of building autonomous AI systems that understand the physical world through learned world models.

---

### 4. LLM-JEPA — JEPA for Language Models

*"LLM-JEPA: Self-Supervised Predictive Learning for Language Models"*

![LLM-JEPA Architecture](./architecture/LLM-JEPA.png)

**Architecture Breakdown:**

LLM-JEPA brings the JEPA paradigm to natural language, combining the standard autoregressive next-token prediction objective with a JEPA objective. The hybrid approach aims to improve abstract reasoning capabilities that pure next-token prediction struggles with.

**Dual Objective Design:**
- **Next-token prediction (standard LLM objective)** — Preserves the model's generative capabilities and language fluency
- **JEPA objective (embedding-space prediction)** — Adds a parallel objective where the model must predict *embeddings* of masked text regions rather than discrete tokens. This forces the model to learn continuous representations of meaning, unconstrained by vocabulary size or tokenization specifics

**Key difference from BERT-style masking:** BERT masks tokens and predicts the original discrete tokens. LLM-JEPA masks regions and predicts their *continuous embeddings* in a learned latent space. This is a fundamentally different objective — predicting what something *means* vs. predicting what *word* was there.

**Results:** Consistent improvements in fine-tuning accuracy across multiple model families (Llama3, Gemma2, OpenELM, Olmo) and benchmarks (NL-RX, GSM8K, Spider). Notably improves reasoning tasks (mathematical, logical) and shows increased robustness against overfitting.

**Limitation:** Training cost is approximately 3x higher due to separate forward passes for each view — motivating ongoing research into more efficient implementations using masked attention.

---

## Connecting the Dots — JEPA Evolution

```
I-JEPA (2023)          Predict image patch representations from context
    |                  → Proved latent prediction works for images
    v
V-JEPA (2024)          Extend to video with spatio-temporal masking
    |                  → Showed temporal reasoning emerges naturally  
    v
V-JEPA 2 (2025)        Scale to 1M+ hours, add planning/simulation
    |                  → Demonstrated world model capabilities
    v
LLM-JEPA               Bring JEPA objectives to language models
                       → Improved abstract reasoning in LLMs
```

The common thread: **prediction in representation space produces better features than prediction in input space**, regardless of the modality (images, video, or language).

---

## Papers

| Paper | Reference |
|-------|-----------|
| [I-JEPA paper](./I-JEPA%20paper.pdf) | Assran et al., 2023 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* |
| [V-JEPA paper](./V-JEPA%20paper.pdf) | Bardes et al., 2024 — *V-JEPA: Latent Video Prediction for Visual Representation Learning* |
| [V-JEPA 2 paper](./V-JEPA%202%20paper.pdf) | Assran et al., 2025 — *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* |
| [LLM-JEPA paper](./llm-JEPA%20paper.pdf) | *LLM-JEPA: Self-Supervised Predictive Learning for Language Models* |

---

## Folder Structure

```
JEPA/
├── README.md
├── architecture/
│   ├── I-JEPA.png              # I-JEPA architecture diagram
│   ├── V-JEPA.png              # V-JEPA architecture diagram
│   ├── V-JEPA-2.png            # V-JEPA 2 architecture diagram
│   └── LLM-JEPA.png           # LLM-JEPA architecture diagram
├── I-JEPA paper.pdf
├── V-JEPA paper.pdf
├── V-JEPA 2 paper.pdf
└── llm-JEPA paper.pdf
```
