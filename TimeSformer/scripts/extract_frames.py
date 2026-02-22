"""
Extract Frames from Downloaded Kinetics-400 Videos
====================================================
Walks through data/videos/ and extracts frames from every .mp4 file,
saving them in a structured directory under data/frames/.

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --fps 1          # 1 frame per second
    python scripts/extract_frames.py --fps 8          # 8 frames per second
    python scripts/extract_frames.py --size 224 224   # Resize frames to 224x224

Output Structure:
    data/frames/
    ├── bench_pressing/
    │   ├── <video_name>/
    │   │   ├── frame_0001.jpg
    │   │   ├── frame_0002.jpg
    │   │   └── ...
    │   └── ...
    ├── deadlifting/
    └── pull_ups/
"""

import os
import argparse
import cv2
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


# ──────────────────────────────────────────────────────────────────────
# Frame Extraction
# ──────────────────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float = None,
    resize: tuple = None,
    img_format: str = "jpg",
    quality: int = 95
) -> dict:
    """
    Extract frames from a single video file.

    Args:
        video_path:  Path to the .mp4 video
        output_dir:  Directory to save extracted frames
        target_fps:  If set, extract at this FPS (None = all frames)
        resize:      If set, resize frames to (width, height)
        img_format:  Image format ('jpg' or 'png')
        quality:     JPEG quality (1-100, only for jpg)

    Returns:
        dict with extraction stats
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {"status": "error", "message": f"Cannot open {video_path.name}"}

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame sampling interval
    if target_fps and target_fps < video_fps:
        frame_interval = int(video_fps / target_fps)
    else:
        frame_interval = 1  # Extract every frame

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # JPEG encode params
    if img_format == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    else:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save frames at the target interval
        if frame_idx % frame_interval == 0:
            saved_count += 1

            # Resize if requested
            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            # Save frame
            frame_filename = f"frame_{saved_count:04d}{ext}"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, encode_params)

        frame_idx += 1

    cap.release()

    return {
        "status": "success",
        "video_name": video_path.stem,
        "video_fps": round(video_fps, 2),
        "video_resolution": f"{width}x{height}",
        "video_duration": round(duration, 2),
        "total_frames": total_frames,
        "extracted_frames": saved_count,
        "effective_fps": round(video_fps / frame_interval, 2),
        "output_dir": str(output_dir),
    }


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from downloaded Kinetics-400 videos"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Target FPS for extraction (default: all frames). E.g., --fps 1 for 1 frame/sec"
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=None, metavar=("W", "H"),
        help="Resize frames to WxH (default: original size). E.g., --size 224 224"
    )
    parser.add_argument(
        "--format", choices=["jpg", "png"], default="jpg",
        help="Image format for saved frames (default: jpg)"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--videos_dir", type=str, default=None,
        help=f"Custom videos directory (default: {VIDEOS_DIR})"
    )
    parser.add_argument(
        "--frames_dir", type=str, default=None,
        help=f"Custom output frames directory (default: {FRAMES_DIR})"
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir) if args.videos_dir else VIDEOS_DIR
    frames_dir = Path(args.frames_dir) if args.frames_dir else FRAMES_DIR
    resize = tuple(args.size) if args.size else None

    print("=" * 60)
    print("  Kinetics-400 Frame Extractor")
    print("=" * 60)
    print(f"  Videos dir   : {videos_dir}")
    print(f"  Frames dir   : {frames_dir}")
    print(f"  Target FPS   : {args.fps or 'all frames'}")
    print(f"  Resize       : {f'{resize[0]}x{resize[1]}' if resize else 'original'}")
    print(f"  Format       : {args.format}")
    print("=" * 60)

    if not videos_dir.exists():
        print(f"\n  ✗ Videos directory not found: {videos_dir}")
        print("  Run download_videos.py first!")
        return

    # Discover all class folders and videos
    class_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"\n  ✗ No class folders found in {videos_dir}")
        return

    total_videos = 0
    total_frames = 0
    all_results = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        videos = sorted(class_dir.glob("*.mp4"))

        if not videos:
            print(f"\n  ⚠ No .mp4 files in {class_name}/")
            continue

        print(f"\n  📁 {class_name} ({len(videos)} videos)")

        for video_path in videos:
            video_name = video_path.stem
            output_dir = frames_dir / class_name / video_name

            # Skip if frames already extracted
            if output_dir.exists() and any(output_dir.iterdir()):
                existing = len(list(output_dir.glob(f"*.{args.format}")))
                print(f"    ✓ Already extracted: {video_name} ({existing} frames)")
                total_videos += 1
                total_frames += existing
                continue

            result = extract_frames_from_video(
                video_path=video_path,
                output_dir=output_dir,
                target_fps=args.fps,
                resize=resize,
                img_format=args.format,
                quality=args.quality,
            )

            if result["status"] == "success":
                print(
                    f"    ✓ {video_name}: {result['extracted_frames']} frames "
                    f"({result['video_duration']}s @ {result['effective_fps']} fps, "
                    f"{result['video_resolution']})"
                )
                total_videos += 1
                total_frames += result["extracted_frames"]
                all_results.append(result)
            else:
                print(f"    ✗ {result.get('message', 'Unknown error')}")

    # Summary
    print("\n" + "=" * 60)
    print("  Extraction Summary")
    print("=" * 60)
    print(f"  Videos processed   : {total_videos}")
    print(f"  Total frames       : {total_frames}")
    print(f"  Output directory   : {frames_dir}")
    print("=" * 60)

    # Per-class breakdown
    if frames_dir.exists():
        for class_dir in sorted(frames_dir.iterdir()):
            if class_dir.is_dir():
                video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
                total_class_frames = sum(
                    len(list(vd.glob(f"*.{args.format}"))) for vd in video_dirs
                )
                print(f"\n  {class_dir.name}:")
                for vd in video_dirs:
                    n_frames = len(list(vd.glob(f"*.{args.format}")))
                    print(f"    → {vd.name}: {n_frames} frames")
                print(f"    Total: {total_class_frames} frames")


if __name__ == "__main__":
    main()
