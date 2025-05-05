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
from data.roboTwin import RobotTwinDataset_Goalgen
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from utils.pipeline import Pipeline
import argparse
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms


def psnr(input_img, target_img):
    metric = PeakSignalNoiseRatio()
    return metric(input_img, target_img)


def ssim(input_img, target_img):
    metric = StructuralSimilarityIndexMeasure()
    return metric(input_img, target_img)


class IP2PEvaluation(object):
    def __init__(self, config):
        # Init models
        pretrained_model_dir = config["pretrained_model_dir"]
        encoder_dir = config["encoder_dir"]
        device = config["device"]
        self.config = config

        self.tokenizer = T5Tokenizer.from_pretrained(encoder_dir, local_files_only=True)
        self.text_encoder = T5EncoderModel.from_pretrained(
            encoder_dir, local_files_only=True
        )
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_dir, subfolder="vae")

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_dir, subfolder="unet"
        )
        payload = torch.load(config["ckpt_path"])
        unet.load_state_dict(payload["state_dict"]["unet_ema"], strict=True)
        self.unet = unet.to(dtype=torch.bfloat16)

        print("happy")

        del payload
        gc.collect()
        torch.cuda.empty_cache()

        self.pipe = Pipeline.from_pretrained(
            pretrained_model_dir,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            vae=self.vae,
            unet=self.unet,
            revision=None,
            variant=None,
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.eval_result_dir = os.path.join(config["result_root"], config["exp_name"])

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.generator = torch.Generator(device).manual_seed(config["seed"])

        # Diffusion hyparams
        self.num_inference_steps = 50
        self.image_guidance_scale = 2.5
        self.guidance_scale = 7.5

        # Image transform
        self.res = config["resolution"]
        self.transform = transforms.Resize((self.res, self.res))

    def evaluate(self):
        os.makedirs(self.eval_result_dir, exist_ok=True)

        dataset = RobotTwinDataset_Goalgen(
            ori_data_dir=self.config["ori_data_dir"],
            data_dir=self.config["data_dir"],
            resolution=256,
            resolution_before_crop=288,
            center_crop=True,
            forward_n_min_max=(5, 5),
            split_type="test",
            use_full=False,
            color_aug=False,
        )

        all_ssim_values = []
        all_psnr_values = []
        sample_metrics = []
        for i in tqdm(len(dataset)):
            example = dataset[i]
            text = example["input_text"]
            original_pixel_values = example["original_pixel_values"]
            edited_pixel_values = example["edited_pixel_values"]
            input_image_batch = [original_pixel_values]
            predict_image = self.inference(input_image_batch, text)

            predict_tensor = (
                torch.tensor(predict_image).permute(2, 0, 1).unsqueeze(0).float()
                / 255.0
            )
            edited_tensor = edited_pixel_values.unsqueeze(0)

            ssim_value = ssim(predict_tensor, edited_tensor)
            psnr_value = psnr(predict_tensor, edited_tensor)
            all_ssim_values.append(ssim_value.item())
            all_psnr_values.append(psnr_value.item())

            sample_metrics.append(
                {"Sample_ID": i, "SSIM": ssim_value.item(), "PSNR": psnr_value.item()}
            )

            _, ax = plt.subplots(1, 3)
            original_image = original_pixel_values.permute(1, 2, 0).numpy()
            original_image = (original_image + 1) / 2 * 255
            original_image = np.clip(original_image, 0, 255)
            original_image = original_image.astype(np.uint8)
            ax[0].imshow(original_image)

            edited_image = edited_pixel_values.permute(1, 2, 0).numpy()
            edited_image = (edited_image + 1) / 2 * 255
            edited_image = np.clip(edited_image, 0, 255)
            edited_image = edited_image.astype(np.uint8)
            ax[1].imshow(edited_image)

            ax[2].imshow(predict_image[0])

            save_dir = os.path.join(
                self.eval_result_dir, example["episode"], f"{example['id']}_debug.png"
            )
            plt.savefig(save_dir, dpi=300)
            plt.close()

        df = pd.DataFrame(sample_metrics)
        df.to_csv(os.path.join(self.eval_result_dir, "metrics.csv"), index=False)

        avg_ssim = sum(all_ssim_values) / len(all_ssim_values) if all_ssim_values else 0
        avg_psnr = sum(all_psnr_values) / len(all_psnr_values) if all_psnr_values else 0
        print(
            f"Evaluation complete. Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f}dB"
        )

        with open(os.path.join(self.eval_result_dir, "summary.txt"), "w") as f:
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}dB\n")
            f.write(f"Number of Samples: {len(all_ssim_values)}\n")

        self.plot_metrics_distribution(all_ssim_values, all_psnr_values)

    def inference(self, image_batch, text_batch):
        """Inference function."""
        input_images = []
        for image in image_batch:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            input_image = self.transform(image)
            input_images.append(input_image)
        edited_images = self.pipe(
            prompt=text_batch,
            image=input_images,
            num_inference_steps=50,
            image_guidance_scale=2.5,
            guidance_scale=7.5,
            generator=self.generator,
            safety_checker=None,
            requires_safety_checker=False,
        ).images
        edited_images = [np.array(image) for image in edited_images]

        return edited_images

    def plot_metrics_distribution(self, ssim_values, psnr_values):
        plt.figure(figsize=(12, 5))

        # SSIM distribution
        plt.subplot(1, 2, 1)
        plt.hist(ssim_values, bins=20, color="blue", alpha=0.7)
        plt.axvline(
            np.mean(ssim_values),
            color="red",
            linestyle="dashed",
            label=f"Average: {np.mean(ssim_values):.4f}",
        )
        plt.title("SSIM Distribution")
        plt.xlabel("SSIM Value")
        plt.ylabel("Sample Count")
        plt.legend()

        # PSNR distribution
        plt.subplot(1, 2, 2)
        plt.hist(psnr_values, bins=20, color="green", alpha=0.7)
        plt.axvline(
            np.mean(psnr_values),
            color="red",
            linestyle="dashed",
            label=f"Average: {np.mean(psnr_values):.2f} dB",
        )
        plt.title("PSNR Distribution")
        plt.xlabel("PSNR Value (dB)")
        plt.ylabel("Sample Count")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.eval_result_dir, "metrics_distribution.png"), dpi=300
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = json.load(open(args.config))

    eval = IP2PEvaluation(config)
    eval.evaluate()
