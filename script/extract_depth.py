from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import glob
import numpy as np
import os
import pickle


def process_pkl_file(pkl_path):
    """处理单个PKL文件，提取深度数据并返回结构化信息"""
    try:
        # 从文件路径中提取任务名称、episode和索引信息
        file_name = os.path.basename(pkl_path)
        index = int(file_name.split(".")[0])
        episode_dir = os.path.basename(os.path.dirname(pkl_path))
        episode_num = int(episode_dir.replace("episode", ""))

        # 获取任务名称
        task_dir = os.path.dirname(os.path.dirname(pkl_path))
        task_name = os.path.basename(task_dir).replace("_pkl", "")

        # 加载PKL文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 提取每个相机的深度数据
        result = {
            "task": task_name,
            "episode": episode_num,
            "index": index,
            "cameras": {},
        }

        cameras = ["head_camera", "front_camera", "left_camera", "right_camera"]

        for cam_name in cameras:
            if (
                cam_name in data["observation"]
                and "depth" in data["observation"][cam_name]
            ):
                # 直接使用numpy数组，不转换为列表
                depth = data["observation"][cam_name]["depth"]
                result["cameras"][cam_name] = depth

        return result
    except Exception as e:
        print(f"Error processing {pkl_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="提取RoboTwin深度数据并按任务和摄像头组织"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="包含PKL文件的数据目录"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出数据目录")
    parser.add_argument("--num_workers", type=int, default=8, help="并行工作进程数")
    args = parser.parse_args()

    # 查找所有任务目录
    task_dirs = glob.glob(os.path.join(args.input_dir, "*_pkl"))
    all_pkl_files = []

    print(f"找到 {len(task_dirs)} 个任务目录")

    # 收集所有PKL文件路径和相关信息
    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir).replace("_pkl", "")
        task_pkls = glob.glob(os.path.join(task_dir, "episode*", "*.pkl"))
        all_pkl_files.extend(task_pkls)
        print(f"任务 '{task_name}' 包含 {len(task_pkls)} 个PKL文件")

    print(f"总共找到 {len(all_pkl_files)} 个PKL文件需要处理")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 并行处理所有PKL文件
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_path = {
            executor.submit(process_pkl_file, path): path for path in all_pkl_files
        }

        # 显示进度条
        for future in tqdm(
            future_to_path, total=len(all_pkl_files), desc="提取深度数据"
        ):
            result = future.result()
            if result:
                results.append(result)

    print(f"成功处理 {len(results)}/{len(all_pkl_files)} 个文件")

    # 按任务、相机类型组织数据，并保持episode和index的顺序
    organized_data = defaultdict(lambda: defaultdict(dict))

    # 先按任务名和相机类型组织
    for result in results:
        task = result["task"]
        episode = result["episode"]
        index = result["index"]

        # 确保每个episode的字典存在
        for cam_name, depth_data in result["cameras"].items():
            if episode not in organized_data[task][cam_name]:
                organized_data[task][cam_name][episode] = {}
            organized_data[task][cam_name][episode][index] = depth_data

        for task, cameras in organized_data.items():
            task_dir = os.path.join(args.output_dir, task)
            os.makedirs(task_dir, exist_ok=True)

            for camera, episodes in cameras.items():
                # 为每个摄像头创建一个压缩的npz文件
                camera_file = os.path.join(task_dir, f"{camera}_depth.npz")

                # 准备要保存的数组，用字典保存
                arrays_dict = {}

                # 遍历每个episode和index，构建数组名称
                for episode in sorted(episodes.keys()):
                    for index in sorted(episodes[episode].keys()):
                        array_name = f"ep{episode}_idx{index}"
                        arrays_dict[array_name] = episodes[episode][index]

                # 保存为压缩的npz文件
                print(f"保存 {task}/{camera} 的深度数据...")
                np.savez_compressed(camera_file, **arrays_dict)

    print("\n数据保存完成!")
    print("按任务和相机类型保存的深度数据目录结构:")
    for task in organized_data.keys():
        print(f"{task}/")
        for camera in organized_data[task].keys():
            print(f"{camera}_depth.json")


if __name__ == "__main__":
    main()
