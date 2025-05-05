#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm
import argparse
import glob
import numpy as np
import os
import pickle


def process_pkl_file(pkl_path, output_base_dir):
    """Process a single PKL file, extract RGB images classified by task and camera type"""
    try:
        # Extract task, episode, and index information from the PKL file path
        file_name = os.path.basename(pkl_path)
        index = file_name.split(".")[0]
        episode_dir = os.path.basename(os.path.dirname(pkl_path))

        # Get task name (directory format is task_name_pkl)
        task_dir = os.path.dirname(os.path.dirname(pkl_path))
        task_name = os.path.basename(task_dir).replace("_pkl", "")

        # Create output directory structure: task_name/camera_type/episode/
        camera_dirs = {
            "head": os.path.join(
                output_base_dir, task_name, "head_camera", episode_dir
            ),
            "front": os.path.join(
                output_base_dir, task_name, "front_camera", episode_dir
            ),
            "left": os.path.join(
                output_base_dir, task_name, "left_camera", episode_dir
            ),
            "right": os.path.join(
                output_base_dir, task_name, "right_camera", episode_dir
            ),
        }

        for d in camera_dirs.values():
            os.makedirs(d, exist_ok=True)

        # Load PKL file
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Save RGB images from all cameras
        for cam_name, dir_key in zip(
            ["head_camera", "front_camera", "left_camera", "right_camera"],
            ["head", "front", "left", "right"],
        ):
            if (
                cam_name in data["observation"]
                and "rgb" in data["observation"][cam_name]
            ):
                rgb = data["observation"][cam_name]["rgb"]
                output_path = os.path.join(camera_dirs[dir_key], f"{index}.png")

                # Check if it's a valid RGB image
                if rgb.shape[2] == 3:
                    # Convert to PIL image and save
                    if np.max(rgb) <= 1.0:
                        # Range [0,1], convert to [0,255]
                        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
                        img = Image.fromarray(rgb_uint8)
                    else:
                        # Already in uint8 format [0,255]
                        img = Image.fromarray(rgb.astype(np.uint8))

                    img.save(output_path)

        return True
    except Exception as e:
        print(f"Error processing {pkl_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract RGB images from RoboTwin PKL files by task and camera view"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Data directory containing PKL files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for RGB images"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel worker processes"
    )
    args = parser.parse_args()

    # Find all PKL files, classified by task
    task_dirs = glob.glob(os.path.join(args.input_dir, "*_pkl"))
    pkl_files = []

    print(f"Found {len(task_dirs)} task directories")

    for task_dir in task_dirs:
        task_pkls = glob.glob(os.path.join(task_dir, "episode*", "*.pkl"))
        pkl_files.extend(task_pkls)
        task_name = os.path.basename(task_dir).replace("_pkl", "")
        print(f"Task '{task_name}' contains {len(task_pkls)} PKL files")

    print(f"Total of {len(pkl_files)} PKL files to process")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all PKL files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_pkl_file, f, args.output_dir) for f in pkl_files
        ]

        # Show progress bar
        success_count = 0
        for future in tqdm(futures, total=len(pkl_files), desc="Extracting images"):
            if future.result():
                success_count += 1

    print(f"Complete! Successfully processed {success_count}/{len(pkl_files)} files")


if __name__ == "__main__":
    main()
