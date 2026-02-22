# TimeSformer -- Is Space-Time Attention All You Need for Video Understanding?

**Paper:** [TimeSformer (ICML 2021)](https://arxiv.org/abs/2102.05095)
**Authors:** Gedas Bertasius, Heng Wang, Lorenzo Torresani (Facebook AI)

---

## Overview

TimeSformer is the **first convolution-free architecture** for video understanding, built entirely on self-attention. It adapts the Vision Transformer (ViT) to video by applying temporal attention and spatial attention to sequences of frame-level patches -- replacing the 3D convolutions used in prior models like I3D, SlowFast, etc.

### Key Idea: Divided Space-Time Attention

Instead of applying full spatiotemporal attention (which is computationally prohibitive), TimeSformer decomposes attention into:
1. **Temporal Attention** -- each patch attends to the same spatial location across all frames
2. **Spatial Attention** -- each patch attends to all patches within the same frame

This "divided" attention achieves the **best accuracy-compute tradeoff** among five attention schemes tested in the paper.

### Architecture

![TimeSformer Architecture](timeSfomer%20architecture.png)

For the editable architecture diagram, see the [Excalidraw source file](timeSformer%20architecture%20breakdown.excalidraw).

---

## Dataset: Kinetics-400 (Mini Subset)

This folder contains scripts to download a **3-class subset** of the [Kinetics-400](https://deepmind.google/discover/blog/open-sourcing-a-state-of-the-art-model/) dataset:

| Class | Description |
|-------|-------------|
| Bench Pressing | Barbell bench press exercise |
| Deadlifting | Barbell deadlift exercise |
| Pull Ups | Pull-up bar exercise |

---

## Folder Structure

```
TimeSformer/
├── README.md                              # This file
├── timeSfomer architecture.png            # Architecture diagram
├── timeSformer architecture breakdown.excalidraw  # Editable diagram source
├── timeSformer paper.pdf                  # Original research paper
├── data/
│   ├── .gitignore                         # Ignores downloaded data
│   ├── annotations/                       # Kinetics-400 CSVs (auto-downloaded)
│   ├── videos/                            # Downloaded .mp4 clips
│   │   ├── bench_pressing/
│   │   ├── deadlifting/
│   │   └── pull_ups/
│   └── frames/                            # Extracted frames per video
│       ├── bench_pressing/
│       ├── deadlifting/
│       └── pull_ups/
└── scripts/
    ├── download_videos.py                 # Download K400 videos
    └── extract_frames.py                  # Extract frames from videos
```

---

## Setup and Usage

### 1. Install Dependencies

```bash
pip install yt-dlp pandas requests opencv-python
```

### 2. Download Videos

```bash
# Download 5 videos per class (default)
python scripts/download_videos.py

# Download more videos
python scripts/download_videos.py --max_videos_per_class 10

# Download from validation set
python scripts/download_videos.py --split validate

# Download both train + validation
python scripts/download_videos.py --split both

# Custom classes
python scripts/download_videos.py --classes "squat" "push up" "yoga"
```

### 3. Extract Frames

```bash
# Extract all frames (native FPS)
python scripts/extract_frames.py

# Extract at 1 frame per second
python scripts/extract_frames.py --fps 1

# Extract at 8 FPS with 224x224 resize (TimeSformer input size)
python scripts/extract_frames.py --fps 8 --size 224 224

# Save as PNG instead of JPG
python scripts/extract_frames.py --format png
```

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Architecture Diagram | Complete |
| Research Paper | Included |
| Dataset Download Scripts | Complete (Kinetics-400 subset) |
| Frame Extraction Scripts | Complete |
| From-Scratch Implementation | Planned |

---

## Notes

- **YouTube Availability**: Some videos may have been removed from YouTube since the dataset was published. The download script handles this gracefully (logs and skips).
- **yt-dlp Required**: Must have [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed (`pip install yt-dlp`).
- **Storage**: Each 10-second clip is approximately 2-5 MB. Frames at native FPS can be approximately 30 frames/second x 10 seconds = 300 frames per video.

---

## References

| Resource | Link |
|----------|------|
| TimeSformer Paper (arXiv) | [arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095) |
| Kinetics-400 Dataset | [deepmind.google](https://deepmind.google/discover/blog/open-sourcing-a-state-of-the-art-model/) |
| Official Code | [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer) |

---

**Note**: This documentation is part of the transformers-CV repository focusing on computer vision architectures built with transformers.
