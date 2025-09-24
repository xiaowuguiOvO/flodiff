import os
from typing import List, Optional, Dict

import swanlab
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from training.train_utils import train_flona, evaluate_flona

# --- 新增 ---
# 导入 Accelerator 用于类型提示
from accelerate import Accelerator


def train_eval_loop_flona(
    train_model: bool,
    model: nn.Module,
    optimizer: AdamW, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_loader: DataLoader,
    transform: transforms,
    epochs: int,
    accelerator: Accelerator,
    project_folder: str,
    print_log_freq: int = 100,
    swanlab_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_swanlab: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    使用 accelerate 来训练和评估模型
    """
    # ==============================训练==============================
    # EMA 模型应该使用未包装的原始模型来初始化
    ema_model = EMAModel(model=accelerator.unwrap_model(model), power=0.75)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            # --- 修改 ---
            # 使用 accelerator.print 来避免在多卡环境下重复打印日志
            accelerator.print(
                f"开始 Flona 训练 - Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_flona(
                model=model, # 传入由 accelerator 包装过的模型
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                # --- 修改 ---
                # 传入 accelerator 对象，替代 device
                accelerator=accelerator,
                noise_scheduler=noise_scheduler,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                swanlab_log_freq=swanlab_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_swanlab=use_swanlab,
                alpha=alpha,
            )
        
        # --- 新增 ---
        # 使用 barrier (屏障)，确保所有进程都完成了训练步骤再进行保存
        accelerator.wait_for_everyone()

        # --- 修改 ---
        # 所有的保存和日志记录操作都只应由主进程执行
        if accelerator.is_main_process:
            # 使用 unwrap_model 来获取原始模型，而不是分布式包装器
            unwrapped_model = accelerator.unwrap_model(model)

            # 将所有状态整合到一个 checkpoint 文件中，方便管理
            checkpoint_to_save = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'ema_model': ema_model.averaged_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
            
            # 保存整合后的 checkpoint
            latest_path = os.path.join(project_folder, "latest.pth")
            numbered_path = os.path.join(project_folder, f"checkpoint_{epoch}.pth")
            
            torch.save(checkpoint_to_save, latest_path)
            torch.save(checkpoint_to_save, numbered_path)

            accelerator.print(f"已保存 checkpoint 至: {numbered_path}")

        #==============================评估==============================
        if (epoch + 1) % eval_freq == 0: 
            accelerator.print(
                f"开始 Flona 评估 - Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_loader
            evaluate_flona(
                ema_model=ema_model,
                dataloader=loader,
                transform=transform,
                # --- 修改 ---
                # 传入 accelerator 对象，替代 device
                accelerator=accelerator,
                noise_scheduler=noise_scheduler,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                num_images_log=num_images_log,
                swanlab_log_freq=swanlab_log_freq,
                use_swanlab=use_swanlab,
                eval_fraction=eval_fraction,
            )
        
        # 在所有进程上同步更新学习率调度器
        if lr_scheduler is not None:
            lr_scheduler.step()

        # --- 修改 ---
        # 只在主进程上记录日志
        if accelerator.is_main_process and use_swanlab:
            swanlab.log({
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            })
    
    if accelerator.is_main_process and use_swanlab:
        swanlab.log({})
    
    accelerator.print("\n训练和评估循环结束。")


def load_model(model, checkpoint: dict) -> None:
    """从 checkpoint 中加载模型状态。"""
    # 这个函数是正确的，因为它在 accelerator.prepare() 之前被调用，
    # 将权重加载到原始模型中。
    # 为了更稳健，我们优先从 'model' 键加载，如果不存在，则加载整个字典。
    model_state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(model_state_dict, strict=False)