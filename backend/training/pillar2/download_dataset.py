"""
Automated Large-Scale Dataset Downloader for Antigravity Agent.
Downloads up to 400 real & deepfake video files (200 Real, 200 Fake from Celeb-DF/DFDC subset)
directly into test_videos/real and test_videos/deepfake via Hugging Face.
"""

import os
import shutil
import argparse
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "angads24/deepfake-video"


def download_dataset(target_dir: str = "test_videos", max_per_class: int = 50):
    """
    Download video files from repo.
    max_per_class: number of videos per class (e.g. 50 = 100 total videos, 100 = 200 total videos).
    """
    real_out = os.path.join(target_dir, "real")
    fake_out = os.path.join(target_dir, "deepfake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    print("=" * 65)
    print("🚀 Fetching Large Deepfake Dataset Index (angads24/deepfake-video)...")
    print("=" * 65)

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    real_files = [f for f in all_files if "dataset/real/" in f and f.endswith(".mp4")]
    fake_files = [f for f in all_files if "dataset/fake/" in f and f.endswith(".mp4")]

    if max_per_class is not None:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    print(f"\n[Download Queue Summary]")
    print(f"  • Real Videos Selected:     {len(real_files)}")
    print(f"  • Deepfake Videos Selected: {len(fake_files)}")
    print(f"  • Total Videos to Download: {len(real_files) + len(fake_files)}")
    print("=" * 65 + "\n")

    # 1. Download Real Videos
    print(f"[1/2] Downloading Real Videos ({len(real_files)} files)...")
    for idx, repo_path in enumerate(real_files, 1):
        clean_name = os.path.basename(repo_path)
        dest_path = os.path.join(real_out, clean_name)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            print(f"  [{idx:3d}/{len(real_files)}] [✓] Cached: {clean_name}")
            continue

        print(f"  [{idx:3d}/{len(real_files)}] Downloading: {clean_name}...")
        try:
            cached_file = hf_hub_download(repo_id=REPO_ID, filename=repo_path, repo_type="dataset")
            shutil.copy(cached_file, dest_path)
            print(f"        [✓] Saved ({os.path.getsize(dest_path) // 1024} KB)")
        except Exception as e:
            print(f"        [!] Error downloading {clean_name}: {e}")

    # 2. Download Fake Videos
    print(f"\n[2/2] Downloading Deepfake Videos ({len(fake_files)} files)...")
    for idx, repo_path in enumerate(fake_files, 1):
        clean_name = os.path.basename(repo_path)
        dest_path = os.path.join(fake_out, clean_name)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            print(f"  [{idx:3d}/{len(fake_files)}] [✓] Cached: {clean_name}")
            continue

        print(f"  [{idx:3d}/{len(fake_files)}] Downloading: {clean_name}...")
        try:
            cached_file = hf_hub_download(repo_id=REPO_ID, filename=repo_path, repo_type="dataset")
            shutil.copy(cached_file, dest_path)
            print(f"        [✓] Saved ({os.path.getsize(dest_path) // 1024} KB)")
        except Exception as e:
            print(f"        [!] Error downloading {clean_name}: {e}")

    print("\n" + "=" * 65)
    print("✅ Huge Dataset Download Complete!")
    print(f"   Real videos in:     {real_out} (Total files: {len(os.listdir(real_out))})")
    print(f"   Deepfake videos in: {fake_out} (Total files: {len(os.listdir(fake_out))})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download large deepfake video dataset")
    parser.add_argument("--target_dir", type=str, default="test_videos", help="Target output folder")
    parser.add_argument("--limit", type=int, default=50, help="Number of videos per category (default: 50 -> 100 total videos)")
    args = parser.parse_args()

    download_dataset(target_dir=args.target_dir, max_per_class=args.limit)
