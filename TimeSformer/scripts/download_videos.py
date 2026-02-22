"""
Download Kinetics-400 Videos for Selected Classes
===================================================
Downloads a subset of the Kinetics-400 dataset by:
1. Fetching official CSV annotations from DeepMind's GitHub
2. Filtering for target action classes
3. Using yt-dlp to download trimmed YouTube clips

Usage:
    python scripts/download_videos.py
    python scripts/download_videos.py --max_videos_per_class 10
    python scripts/download_videos.py --split train
    python scripts/download_videos.py --classes "bench pressing" "deadlifting" "squat"
"""

import os
import sys
import argparse
import subprocess
import requests
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Official Kinetics-400 annotation URLs (CVDF / S3)
ANNOTATION_URLS = {
    "train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "validate": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
}

# Fallback: direct DeepMind hosted annotations
ANNOTATION_URLS_FALLBACK = {
    "train": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/train.csv",
    "validate": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400/val.csv",
}

# Default target classes (gym/weightlifting theme)
DEFAULT_CLASSES = ["bench pressing", "deadlifting", "pull ups"]

# Project root (relative to this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
VIDEOS_DIR = DATA_DIR / "videos"


# ──────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────

def download_annotations(split: str) -> Path:
    """Download Kinetics-400 CSV annotation file if not already present."""
    csv_path = ANNOTATIONS_DIR / f"{split}.csv"

    if csv_path.exists():
        print(f"  ✓ Annotations already exist: {csv_path.name}")
        return csv_path

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Try primary URL first, then fallback
    for label, urls in [("primary", ANNOTATION_URLS), ("fallback", ANNOTATION_URLS_FALLBACK)]:
        url = urls.get(split)
        if not url:
            continue

        print(f"  ↓ Downloading {split}.csv ({label})...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            csv_path.write_bytes(response.content)
            print(f"  ✓ Saved to {csv_path}")
            return csv_path
        except requests.RequestException as e:
            print(f"  ✗ {label} URL failed: {e}")

    print(f"  ✗ ERROR: Could not download {split} annotations from any source.")
    print(f"    Manual download: Place {split}.csv in {ANNOTATIONS_DIR}")
    sys.exit(1)


def filter_annotations(csv_path: Path, target_classes: list) -> pd.DataFrame:
    """Filter annotation CSV for target classes only."""
    df = pd.read_csv(csv_path)

    # Kinetics CSVs have columns: label, youtube_id, time_start, time_end, split, is_cc
    # Normalize class names for comparison
    df["label_lower"] = df["label"].str.strip().str.lower()
    target_lower = [c.strip().lower() for c in target_classes]

    filtered = df[df["label_lower"].isin(target_lower)].copy()
    filtered.drop(columns=["label_lower"], inplace=True)

    return filtered


def check_ytdlp_installed() -> bool:
    """Check if yt-dlp is installed and accessible."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_video(youtube_id: str, time_start: int, time_end: int,
                   output_path: Path) -> bool:
    """Download a single YouTube clip using yt-dlp."""
    # Skip if already downloaded with content
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"    ✓ Already exists: {output_path.name}")
        return True

    # Clean up any empty/corrupt files from previous attempts
    if output_path.exists() and output_path.stat().st_size <= 1000:
        output_path.unlink()

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Download at 480p (ideal for training — TimeSformer resizes to 224x224)
    # YouTube serves video-only + audio-only streams; ffmpeg merges them
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--socket-timeout", "30",
        url
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )

        # Clean up empty files
        if output_path.exists() and output_path.stat().st_size <= 1000:
            output_path.unlink()

        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
            return True
        else:
            error = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            if "unavailable" in error.lower() or "private" in error.lower() or "removed" in error.lower():
                print(f"    ✗ Unavailable: {youtube_id}")
            else:
                print(f"    ✗ Failed: {youtube_id} — {error}")
            return False

    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout: {youtube_id}")
        return False
    except Exception as e:
        print(f"    ✗ Error: {youtube_id} — {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Kinetics-400 videos for selected action classes"
    )
    parser.add_argument(
        "--classes", nargs="+", default=DEFAULT_CLASSES,
        help=f"Action classes to download (default: {DEFAULT_CLASSES})"
    )
    parser.add_argument(
        "--max_videos_per_class", type=int, default=5,
        help="Max number of videos to download per class (default: 5)"
    )
    parser.add_argument(
        "--split", choices=["train", "validate", "both"], default="train",
        help="Dataset split to download from (default: train)"
    )
    args = parser.parse_args()

    splits = ["train", "validate"] if args.split == "both" else [args.split]
    target_classes = args.classes

    print("=" * 60)
    print("  Kinetics-400 Video Downloader")
    print("=" * 60)
    print(f"  Target classes : {target_classes}")
    print(f"  Max per class  : {args.max_videos_per_class}")
    print(f"  Split(s)       : {splits}")
    print(f"  Output dir     : {VIDEOS_DIR}")
    print("=" * 60)

    # Step 0: Check yt-dlp
    print("\n[0/3] Checking yt-dlp installation...")
    if not check_ytdlp_installed():
        print("  ✗ yt-dlp is not installed!")
        print("  Fix: pip install yt-dlp")
        sys.exit(1)
    print("  ✓ yt-dlp is available")

    # Step 1: Download annotations
    print("\n[1/3] Downloading annotations...")
    all_filtered = []
    for split in splits:
        csv_path = download_annotations(split)
        filtered = filter_annotations(csv_path, target_classes)
        all_filtered.append(filtered)
        print(f"  → Found {len(filtered)} videos for target classes in {split}")

    combined = pd.concat(all_filtered, ignore_index=True)

    if combined.empty:
        print("\n  ✗ No videos found for the specified classes!")
        print(f"  Check class names match exactly. Available in annotations:")
        for split in splits:
            df = pd.read_csv(ANNOTATIONS_DIR / f"{split}.csv")
            classes = sorted(df["label"].unique())
            print(f"  {split}: {len(classes)} classes")
        sys.exit(1)

    # Step 2: Download videos
    print("\n[2/3] Downloading videos...")
    stats = {"success": 0, "failed": 0, "skipped": 0}

    for class_name in target_classes:
        # Normalize folder name (spaces → underscores)
        folder_name = class_name.strip().replace(" ", "_")
        class_dir = VIDEOS_DIR / folder_name
        class_dir.mkdir(parents=True, exist_ok=True)

        class_videos = combined[combined["label"].str.strip().str.lower() == class_name.lower()]
        n_to_download = min(len(class_videos), args.max_videos_per_class)

        print(f"\n  📁 {class_name} ({n_to_download}/{len(class_videos)} available)")

        for idx, (_, row) in enumerate(class_videos.head(n_to_download).iterrows()):
            youtube_id = row["youtube_id"]
            time_start = int(row["time_start"])
            time_end = int(row["time_end"])

            filename = f"{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
            output_path = class_dir / filename

            success = download_video(youtube_id, time_start, time_end, output_path)
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    print(f"  ✓ Successful : {stats['success']}")
    print(f"  ✗ Failed     : {stats['failed']}")
    print(f"  Output       : {VIDEOS_DIR}")
    print("=" * 60)

    # List downloaded files per class
    for class_name in target_classes:
        folder_name = class_name.strip().replace(" ", "_")
        class_dir = VIDEOS_DIR / folder_name
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4"))
            print(f"\n  {class_name}: {len(videos)} videos")
            for v in videos:
                size_mb = v.stat().st_size / (1024 * 1024)
                print(f"    → {v.name} ({size_mb:.1f} MB)")

    if stats["failed"] > 0:
        print(f"\n  ⚠  {stats['failed']} videos failed (likely removed from YouTube)")
        print("  Tip: Increase --max_videos_per_class to compensate")


if __name__ == "__main__":
    main()
