from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm
import argparse
import glob
import numpy as np
import os
import pickle


def process_pkl_file(pkl_path, output_base_dir):
    """处理单个PKL文件，按任务和相机类型分类提取RGB图像"""
    try:
        # 从PKL文件路径中提取任务、episode和索引信息
        file_name = os.path.basename(pkl_path)
        index = file_name.split(".")[0]
        episode_dir = os.path.basename(os.path.dirname(pkl_path))

        # 获取任务名称 (目录格式为 task_name_pkl)
        task_dir = os.path.dirname(os.path.dirname(pkl_path))
        task_name = os.path.basename(task_dir).replace("_pkl", "")

        # 创建输出目录结构: 任务名/相机类型/episode/
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

        # 加载PKL文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 保存所有摄像头的RGB图像
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

                # 检查是否为合法RGB图像
                if rgb.shape[2] == 3:
                    # 转换为PIL图像并保存
                    if np.max(rgb) <= 1.0:
                        # 是[0,1]范围，转换为[0,255]
                        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
                        img = Image.fromarray(rgb_uint8)
                    else:
                        # 已经是uint8格式[0,255]
                        img = Image.fromarray(rgb.astype(np.uint8))

                    img.save(output_path)

        return True
    except Exception as e:
        print(f"Error processing {pkl_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="按任务和摄像头视角提取RoboTwin PKL文件中的RGB图像"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="包含PKL文件的数据目录"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="RGB图像输出目录")
    parser.add_argument("--num_workers", type=int, default=8, help="并行工作进程数")
    args = parser.parse_args()

    # 找到所有PKL文件，并按任务分类
    task_dirs = glob.glob(os.path.join(args.input_dir, "*_pkl"))
    pkl_files = []

    print(f"找到 {len(task_dirs)} 个任务目录")

    for task_dir in task_dirs:
        task_pkls = glob.glob(os.path.join(task_dir, "episode*", "*.pkl"))
        pkl_files.extend(task_pkls)
        task_name = os.path.basename(task_dir).replace("_pkl", "")
        print(f"任务 '{task_name}' 包含 {len(task_pkls)} 个PKL文件")

    print(f"总共找到 {len(pkl_files)} 个PKL文件需要处理")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 并行处理所有PKL文件
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_pkl_file, f, args.output_dir) for f in pkl_files
        ]

        # 显示进度条
        success_count = 0
        for future in tqdm(futures, total=len(pkl_files), desc="提取图像"):
            if future.result():
                success_count += 1

    print(f"完成! 成功处理 {success_count}/{len(pkl_files)} 个文件")


if __name__ == "__main__":
    main()
