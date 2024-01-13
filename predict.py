# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Path, Input

import argparse
import os
from datetime import datetime
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        # Read pretrained weights path from config
        config_path="./configs/prompts/animation.yaml"
        config = OmegaConf.load(config_path)
        weight_dtype=torch.float16
        self.config = config
        self.pipeline = None
        self.weight_dtype = weight_dtype

    def predict(
        self,
        ref_image: Path = Input(description="Reference image to use for generation"),
        pose_video_path: Path = Input(description="Motion sequence to use for generation"),
        width: int = Input(
            description="Width of the image to generate", default=512
        ),
        height: int = Input(
            description="Height of the image to generate", default=784
        ),
        length: int = Input(
            description="Length of the video to generate", default=24, le=120
        ),
        seed: int = Input(
            description="Random seed to use for generation", default=-1
        ),
        cfg: float = Input(
            description="Diffusion coefficient", default=3.5, ge=0, le=10
        ),
        num_inference_steps: int = Input(
            description="Number of steps to run diffusion for", default=25
        ),

    ) -> Path:
        generator = torch.manual_seed(seed)
        if isinstance(ref_image, np.ndarray):
            ref_image = Image.fromarray(ref_image)
        if self.pipeline is None:
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_vae_path,
            ).to("cuda", dtype=self.weight_dtype)

            reference_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_base_model_path,
                subfolder="unet",
            ).to(dtype=self.weight_dtype, device="cuda")

            inference_config_path = self.config.inference_config
            infer_config = OmegaConf.load(inference_config_path)
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                self.config.pretrained_base_model_path,
                self.config.motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=self.weight_dtype, device="cuda")

            pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device="cuda"
            )

            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                self.config.image_encoder_path
            ).to(dtype=self.weight_dtype, device="cuda")
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)

            # load pretrained weights
            denoising_unet.load_state_dict(
                torch.load(self.config.denoising_unet_path, map_location="cpu"),
                strict=False,
            )
            reference_unet.load_state_dict(
                torch.load(self.config.reference_unet_path, map_location="cpu"),
            )
            pose_guider.load_state_dict(
                torch.load(self.config.pose_guider_path, map_location="cpu"),
            )

            pipe = Pose2VideoPipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            pipe = pipe.to("cuda", dtype=self.weight_dtype)
            self.pipeline = pipe

        # open video
        pose_video_path = str(pose_video_path)
        ref_image = Image.open(ref_image).convert("RGB")
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)

        pose_list = []
        pose_tensor_list = []
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for pose_image_pil in pose_images[:length]:
            pose_list.append(pose_image_pil)
            pose_tensor_list.append(pose_transform(pose_image_pil))

        video = self.pipeline(
            ref_image,
            pose_list,
            width=width,
            height=height,
            video_length=length,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=generator,
        ).videos

        ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=length
        )
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        save_dir = f"./output/gradio"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = os.path.join(save_dir, f"{date_str}T{time_str}.mp4")
        save_videos_grid(
            video,
            out_path,
            n_rows=3,
            fps=src_fps,
        )

        torch.cuda.empty_cache()

        return Path(out_path)
