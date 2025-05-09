# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import re
import shutil
import torch
import torchvision.transforms as transforms


def extract_path_info(img_path):
    task_pattern = r".+/([^/]+)/episode(\d+)/(\d+)_([^/]+)\.png"
    match = re.match(task_pattern, img_path)

    if match:
        task = match.group(1)
        episode_num = int(match.group(2))
        frame_num = int(match.group(3))
        camera_type = match.group(4)

        return {
            "task": task,
            "episode_num": episode_num,
            "frame_num": frame_num,
            "camera_type": camera_type,
        }


class RobotTwinDataset_Goalgen(Dataset):
    def __init__(
        self,
        ori_data_dir,
        data_dir,
        resolution=256,
        resolution_before_crop=288,
        center_crop=False,
        forward_n_min_max=[5, 5],
        use_full=True,
        split_type="training",
        color_aug=False,
    ):
        super().__init__()
        self.color_aug = color_aug  # whether to use ColorJitter
        self.center_crop = center_crop  # whether to use CenterCrop
        self.ori_data_dir = ori_data_dir
        self.data_dir = os.path.join(data_dir, split_type)

        self.forward_n_min, self.forward_n_max = forward_n_min_max
        self.use_full = use_full  # whether to use every frame in a trajectory
        self.resolution = resolution
        self.resolution_before_crop = resolution_before_crop

        self.split_type = split_type

        # image preprocessing
        if split_type == "training":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (self.resolution_before_crop, self.resolution_before_crop),
                        antialias=False,
                    ),
                    transforms.CenterCrop(self.resolution)
                    if self.center_crop
                    else transforms.RandomCrop(self.resolution),
                ]
            )
        else:
            self.transform = transforms.Resize(
                (self.resolution, self.resolution),
                antialias=False,
            )

        meta_json = os.path.join(self.data_dir, "meta.json")

        with open(meta_json, "r") as f:
            self.meta = json.load(f)

            task_episode_pairs = []

        for task, config in self.meta.items():
            for episode in config["num_frames"].keys():
                task_episode_pairs.append((task, episode))

        # 2. 重新组织sample_tuples，使相同task的不同episode交错排列
        self.sample_tuples = []

        # 如果想打乱顺序但保持每个task轮流出现
        # 首先按task分组
        task_to_episodes = {}
        for task, episode in task_episode_pairs:
            if task not in task_to_episodes:
                task_to_episodes[task] = []
            task_to_episodes[task].append(episode)

        # 然后对每个task的episodes列表进行随机打乱
        for task in task_to_episodes:
            random.shuffle(task_to_episodes[task])

        # 获取所有task列表并随机打乱顺序
        all_tasks = list(task_to_episodes.keys())
        random.shuffle(all_tasks)

        # 轮流从每个task中取一个episode，直到所有episodes都取完
        max_episodes = max(len(episodes) for episodes in task_to_episodes.values())
        for i in range(max_episodes):
            for task in all_tasks:
                episodes = task_to_episodes[task]
                if i < len(episodes):
                    episode = episodes[i]
                    n_frames = self.meta[task]["num_frames"][episode]

                    # 添加此task-episode下的所有有效frame
                    for frame_id in range(n_frames):
                        if self.use_full:
                            temp_max = 1
                        else:
                            temp_max = self.forward_n_max
                        if (frame_id + temp_max) < n_frames:
                            sample_tuple = (task, episode, frame_id, n_frames)
                            self.sample_tuples.append(sample_tuple)

    def __len__(self):
        return len(self.sample_tuples)

    def __getitem__(self, index):
        if self.split_type != "training":
            np.random.seed(index)
            random.seed(index)

        sample_tuple = self.sample_tuples[index]
        task, episode, frame_id, n_frames = sample_tuple

        # text
        edit_prompt = self.meta[task]["text"]

        fallback_image = Image.new(
            "RGB", (self.resolution, self.resolution), color=(0, 0, 0)
        )
        input_image = fallback_image
        edited_image = fallback_image

        input_image_path = os.path.join(
            self.data_dir, task, episode, f"{frame_id}_front_camera.png"
        )
        try:
            input_image = Image.open(input_image_path).convert("RGB")
        except UnidentifiedImageError:
            info = extract_path_info(input_image_path)
            src_path = os.path.join(
                self.ori_data_dir,
                info["task"],
                info["camera_type"],
                f"episode{info['episode_num']}",
                f"{info['frame_num']}.png",
            )

            shutil.copy(src_path, input_image_path)
            input_image = Image.open(input_image_path).convert("RGB")

        # goal image
        forward_n = random.choice(range(self.forward_n_min, self.forward_n_max + 1))
        edited_frame_id = min(frame_id + forward_n, n_frames - 1)
        assert edited_frame_id < n_frames

        edited_image_path = os.path.join(
            self.data_dir, task, episode, f"{edited_frame_id}_front_camera.png"
        )
        try:
            edited_image = Image.open(edited_image_path).convert("RGB")
        except UnidentifiedImageError:
            info = extract_path_info(edited_image_path)
            src_path = os.path.join(
                self.ori_data_dir,
                info["task"],
                info["camera_type"],
                f"episode{info['episode_num']}",
                f"{info['frame_num']}.png",
            )

            shutil.copy(src_path, edited_image_path)
            edited_image = Image.open(edited_image_path).convert("RGB")

        bright_range = random.uniform(0.8, 1.2)
        contrast_range = random.uniform(0.8, 1.2)
        saturation_range = random.uniform(0.8, 1.2)
        hue_range = random.uniform(-0.04, 0.04)
        if (
            self.split_type == "training" and self.color_aug and random.random() > 0.4
        ):  # apply color jitter with probability of 0.6
            self.color_trans = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=(bright_range, bright_range),
                        contrast=(contrast_range, contrast_range),
                        saturation=(saturation_range, saturation_range),
                        hue=(hue_range, hue_range),
                    ),
                ]
            )
            input_image = self.color_trans(input_image)
            edited_image = self.color_trans(
                edited_image
            )  # apply the same transformation to input image and goal image

        # preprocess_images
        concat_images = np.concatenate(
            [np.array(input_image), np.array(edited_image)], axis=2
        )
        concat_images = torch.from_numpy(concat_images).float()
        concat_images = concat_images.permute(2, 0, 1)
        concat_images = 2 * (concat_images / 255) - 1

        concat_images = self.transform(concat_images)

        input_image, edited_image = concat_images.chunk(2)

        input_image = input_image.reshape(3, self.resolution, self.resolution)
        edited_image = edited_image.reshape(3, self.resolution, self.resolution)

        return {
            "input_text": [edit_prompt],
            "original_pixel_values": input_image,
            "edited_pixel_values": edited_image,
            "task": task,
            "episode": episode,
            "frame_id": frame_id,
        }
