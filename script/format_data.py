#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import os
import random
import re
import shutil
import json
import logging
import multiprocessing as mp
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_builder")

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

# Task configuration
task_config = {
    "block_hammer_beat_D435": {
        "text": "beat the block with the hammer",
    },
    "block_handover_D435": {
        "text": "handover the blocks",
    },
    "blocks_stack_easy_D435": {
        "text": "stack blocks",
    },
}

# Camera types
cameras = ["front_camera", "head_camera", "left_camera", "right_camera"]
# Compiled regex for better performance
episode_pattern = re.compile(r"episode[-_]?(\d+)")


def parse_episode_number(path):
    """Extract episode number from path using optimized regex."""
    match = episode_pattern.search(str(path))
    return int(match.group(1)) if match else -1


def batch_process_episodes(episode_batch, src_dir, dest_dir, task_name):
    """Process a batch of episodes with optimized I/O operations.

    Args:
        episode_batch (list): List of episode numbers to process
        src_dir (str): Source directory path
        dest_dir (str): Destination directory path
        task_name (str): Name of the task being processed

    Returns:
        dict: Result containing task name and frame counts
    """

    # Create all destination directories at once
    dest_dirs = {
        ep_num: Path(dest_dir) / f"episode{ep_num}" for ep_num in episode_batch
    }
    for d in dest_dirs.values():
        d.mkdir(exist_ok=True, parents=True)

    for episode_num in episode_batch:
        episode = f"episode{episode_num}"

        dest_ep_dir = dest_dirs[episode_num]

        for camera in cameras:
            # Build source directory path
            src_img_dir = Path(src_dir) / camera / episode

            if not src_img_dir.exists():
                continue

            try:
                # Use Path.glob for more efficient file listing
                img_files = list(src_img_dir.glob("*.png"))

                # Process and copy files in batches
                for img_file in img_files:
                    # Extract pure numeric part as prefix
                    frame_idx = img_file.stem

                    # If it already contains camera name, remove it
                    if any(cam in frame_idx for cam in cameras):
                        # Keep only the numeric part
                        frame_idx = frame_idx.split("_")[0]

                    # Build new filename, ensure only one camera suffix
                    new_filename = f"{frame_idx}_{camera}.png"

                    # Copy with performance optimization
                    shutil.copy2(img_file, dest_ep_dir / new_filename)

            except (OSError, PermissionError) as e:
                logger.warning(f"Error processing {src_img_dir}: {e}")


def main():
    """Main function to build the RobotTwin dataset."""
    parser = argparse.ArgumentParser(description="RobotTwin Dataset Builder Tool")
    parser.add_argument(
        "--input_dir", required=True, help="RobotTwin_output directory path"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path for formatted dataset output"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of episodes to process in one batch",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=100, help="Maximum number of tasks in queue"
    )
    args = parser.parse_args()

    split_types = ["training", "validation"]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory structure upfront
    for split_type in split_types:
        for task in task_config:
            (output_dir / split_type / task).mkdir(exist_ok=True, parents=True)

    # Step 1: Find and distribute episodes - done once upfront
    for task, config in task_config.items():
        # Direct path construction is faster than glob
        task_path = input_dir / task / "front_camera"
        if not task_path.exists():
            logger.warning(f"Path not found: {task_path}")
            continue

        episode_dirs = [d for d in os.listdir(task_path) if d.startswith("episode")]

        # Extract all available episode numbers
        episode_nums = [parse_episode_number(d) for d in episode_dirs]
        episode_nums = [num for num in episode_nums if num != -1]

        # Randomly shuffle episode numbers
        random.shuffle(episode_nums)

        # Split into training and validation sets
        train_count = int(len(episode_nums) * args.train_ratio)
        train_episodes = episode_nums[:train_count]
        val_episodes = episode_nums[train_count:]

        logger.info(
            f"Task {task}: {len(episode_nums)} episodes (Training: {len(train_episodes)}, Validation: {len(val_episodes)})"
        )

        config["training"] = train_episodes
        config["validation"] = val_episodes

    # Step 2: Process in optimized batches
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = []

        for split_type in split_types:
            for task, config in task_config.items():
                episode_nums = config.get(split_type, [])
                if not episode_nums:
                    continue

                src_dir = input_dir / task
                dest_dir = output_dir / split_type / task

                # Process episodes in batches for better I/O efficiency
                for i in range(0, len(episode_nums), args.batch_size):
                    batch = episode_nums[i : i + args.batch_size]
                    futures.append(
                        executor.submit(
                            batch_process_episodes,
                            batch,
                            str(src_dir),
                            str(dest_dir),
                            task,
                        )
                    )

                    # Control memory usage by limiting queue size
                    if len(futures) >= args.buffer_size:
                        process_completed_futures(futures[: args.buffer_size // 2])
                        futures = futures[args.buffer_size // 2 :]

        # Process remaining futures
        process_completed_futures(futures, show_progress=True)

    # Write metadata files
    for split_type in split_types:
        metadata_file = output_dir / split_type / "meta.json"
        with open(metadata_file, "w") as f:
            json.dump(task_config, f, indent=2)

    logger.info(f"Dataset build complete, saved to: {args.output_dir}")


def process_completed_futures(futures, show_progress=False):
    """Process completed futures and update task configurations.

    Args:
        futures (list): List of futures to process
        show_progress (bool): Whether to show a progress bar
    """
    iterator = as_completed(futures)
    if show_progress:
        iterator = tqdm(iterator, total=len(futures), desc="Processing episodes")

    for future in iterator:
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error processing batch: {e}")


if __name__ == "__main__":
    main()
