import os
import argparse
import yaml
import time

import numpy as np
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# --- 1. 导入 Accelerator ---
from accelerate import Accelerator

from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.flona_dataset import flona_Dataset
from train_eval_loop import train_eval_loop_flona, load_model


def main(config):
    # --- 2. 初始化 Accelerator ---
    # 这应该是函数的第一行，它会自动处理所有设备设置
    accelerator = Accelerator()

    # ==============================Dataset==============================
    # 数据集部分代码完全不用变
    data_config = config["datasets"]
    train_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "train"),
        trav_folder=data_config["trav_map_folder"],
        scene_names=data_config['scene_names'],
        image_size=config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
        load_index=config["load_index"]
    )
    test_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "test"),
        trav_folder=data_config["trav_map_folder"],
        scene_names=data_config['scene_names'],
        image_size=config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # ==============================Model==============================
    # 模型定义部分完全不用变
    vision_encoder = flona_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = flona(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # ==============================Training Configuration==============================
    
    # --- 3. 删除了所有手动的设备管理代码 ---
    # 之前关于 os.environ, device, DataParallel 的代码块已被完全移除
    # accelerator 会自动处理这一切
    
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)
    lr = float(config["lr"])
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
    if config["warmup"]:
        print("Using warmup")
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config["warmup_epochs"],
            after_scheduler=scheduler,
        )
    current_epoch = 0

    # checkpoint 加载逻辑，需要在 prepare 之前完成对 model 的加载
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        accelerator.print(f"Loading model from {load_project_folder}")
        latest_path = os.path.join(load_project_folder, "latest.pth")
        
        # 加载时指定 map_location，以防 checkpoint 是在不同设备上保存的
        latest_checkpoint = torch.load(latest_path, map_location='cpu') 
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # --- 4. 使用 accelerator.prepare() 包装所有核心组件 ---
    # 这是最关键的一步，它会自动处理模型和数据的设备移动
    model, optimizer, scheduler, train_loader, test_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, test_dataloader
    )
    
    # 在 prepare 之后加载 optimizer 和 scheduler 的状态
    if "load_run" in config:  
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"])
            
    # ==============================Train==============================
    # --- 5. 修改 train_eval_loop_flona 的调用 ---
    # 删除了 device 参数，并传入了 accelerator
    train_eval_loop_flona(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_loader=test_dataloader,
        transform=transform,
        epochs=config["epochs"],
        accelerator=accelerator,  # <--- 传入 accelerator
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        swanlab_log_freq=config["swanlab_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_swanlab=config["use_swanlab"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )

    accelerator.print("Done!!!")


if __name__ == "__main__":
    # 这部分代码完全不用变
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    parser.add_argument(
        "--config",
        "-c",
        default="flona.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config["project_folder"],
    )
    if config["use_swanlab"]:
        swanlab.login()
        swanlab.init(
            project=config["project_name"],
            settings=swanlab.Settings(start_method="thread"),
        )
        swanlab.run.name = config["run_name"]
        if swanlab.run:
            swanlab.config.update(config)

    main(config)