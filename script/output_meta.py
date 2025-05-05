from pathlib import Path
from tqdm import tqdm
import argparse
import json
import random

seed = 42
random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="RobotTwin Dataset Meta Json Writer")
    parser.add_argument(
        "--input_dir", required=True, help="RobotTwin_format directory path"
    )
    parser.add_argument(
        "--train_samples", type=int, default=32, help="每个任务抽取的训练集episode数量"
    )
    parser.add_argument(
        "--val_samples", type=int, default=8, help="每个任务抽取的验证集episode数量"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    split_types = ["training", "validation"]
    for split_type in split_types:
        split_dir = input_dir / split_type
        task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        task_config = {
            "block_hammer_beat_D435": {
                "text": "beat the block with the hammer",
                "num_frames": {},
            },
            "block_handover_D435": {
                "text": "handover the blocks",
                "num_frames": {},
            },
            "blocks_stack_easy_D435": {
                "text": "stack blocks",
                "num_frames": {},
            },
        }

        for task_dir in task_dirs:
            task_name = task_dir.name

            episode_dirs = [
                d
                for d in task_dir.iterdir()
                if d.is_dir() and d.name.startswith("episode")
            ]

            num_samples = (
                args.train_samples if split_type == "training" else args.val_samples
            )
            sampled_episodes = random.sample(episode_dirs, num_samples)
            for episode_dir in tqdm(sampled_episodes, desc=f"{task_name} episodes"):
                episode_name = episode_dir.name

                files = list(episode_dir.glob("*.png"))
                file_count = len(files) // 4

                task_config[task_name]["num_frames"][episode_name] = file_count

        meta_json = split_dir / "meta.json"
        with open(meta_json, "w") as f:
            json.dump(task_config, f, indent=2)


if __name__ == "__main__":
    main()
