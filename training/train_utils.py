import os
import itertools
from typing import Optional

import tqdm
import numpy as np
import swanlab
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 新增 ---
from accelerate import Accelerator

from training.Logger import Logger
from model.data_utils import VISUALIZATION_IMAGE_SIZE
from model.data_utils import to_global_coords, to_local_coords, img_path_to_data_and_point_transfer

# --- 全局常量保持不变 ---
ACTION_STATS = {}
# --- 修改：在创建 numpy 数组时，直接指定数据类型为 float32 ---
ACTION_STATS["min"] = np.array([-2.5, -4], dtype=np.float32)
ACTION_STATS["max"] = np.array([5, 4], dtype=np.float32)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])


def _compute_losses_flona(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_floorplan_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    # --- 修改: 接收 accelerator ---
    accelerator: Accelerator,
):
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]
    num_samples = 1
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_floorplan_images,
        pred_horizon,
        action_dim,
        num_samples,
        goal_pos,
        curr_pos,
        curr_ori,
        # --- 修改: 传递 accelerator ---
        accelerator=accelerator,
    )
    actions = model_output_dict['actions']
    distance = model_output_dict['distance']

    dist_loss = F.mse_loss(distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    action_loss = action_reduce(F.mse_loss(actions, batch_action_label, reduction="none"))
    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(actions[:, :, :2], batch_action_label[:, :, :2], dim=-1))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(torch.flatten(actions[:, :, :2], start_dim=1), torch.flatten(batch_action_label[:, :, :2], start_dim=1), dim=-1))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        "dist_loss": dist_loss,
    }
    return results


def train_flona(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: AdamW,
    dataloader: DataLoader,
    transform: transforms,
    # --- 修改: 接收 accelerator, 移除 device ---
    accelerator: Accelerator,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    swanlab_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_swanlab: bool = True,
):
    model.train()
    num_batches = len(dataloader)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", "train", window_size=print_log_freq)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    loggers = {"action_loss": action_loss_logger, "action_waypts_cos_sim": action_waypts_cos_sim_logger, "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger, "dist_loss": dist_loss_logger}
    
    # tqdm 进度条只在主进程显示
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False, disable=not accelerator.is_main_process) as tepoch:
        for i, data in enumerate(tepoch):
            # accelerator 会自动处理数据到设备的移动, 无需 .to(device)
            (obs_image, floorplan_image, actions, distance, goal_pos, curr_pos, curr_ori,
             goal_pos_resized, curr_pos_resized, curr_ori_resized, goal_pos_local, curr_pos_local) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_floorplan_images = TF.resize(floorplan_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1)
            batch_floorplan_images = transform(floorplan_image)

            h0, w0 = obs_images[0].shape[-2:]
            h1, w1 = batch_viz_obs_images.shape[-2:]
            
            # --- 修改: 确保在这里创建的 tensor 也在正确的设备上 ---
            device = accelerator.device
            goal_pos_resized = goal_pos_resized * torch.tensor([w1/w0, h1/h0], device=device)
            curr_pos_resized = curr_pos_resized * torch.tensor([w1/w0, h1/h0], device=device)
            curr_ori_resized = curr_ori_resized * torch.tensor([w1/w0, h1/h0], device=device)
            for b in range(goal_pos_resized.shape[0]):
                batch_viz_floorplan_images[b,:,int(goal_pos_resized[b,1]), int(goal_pos_resized[b,0])] = torch.tensor(RED, device=device)
                batch_viz_floorplan_images[b,:,int(curr_pos_resized[b,1]), int(curr_pos_resized[b,0])] = torch.tensor(GREEN, device=device)
                batch_viz_floorplan_images[b,:,int(curr_ori_resized[b,1]), int(curr_ori_resized[b,0])] = torch.tensor(CYAN, device=device)

            B = actions.shape[0]
            obsfloorplan_cond = model("vision_encoder", obs_img=batch_obs_images, floorplan_img=batch_floorplan_images, obs_pos=curr_pos, goal_pos=goal_pos, obs_ori=curr_ori)
            distance = distance.float()
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = ndeltas

            dist_pred = model("dist_pred_net", obsgoal_cond=obsfloorplan_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsfloorplan_cond)
            
            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()
            
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            optimizer.zero_grad()
            # --- 修改: 使用 accelerator.backward() ---
            accelerator.backward(loss)
            optimizer.step()
            ema_model.step(model)

            loss_cpu = loss.item()
            if accelerator.is_main_process:
                tepoch.set_postfix(loss=loss_cpu)
                if use_swanlab:
                    swanlab.log({"total_loss": loss_cpu})
                    swanlab.log({"dist_loss": dist_loss.item()})
                    swanlab.log({"diffusion_loss": diffusion_loss.item()})

            if i % print_log_freq == 0:
                with torch.no_grad():
                    losses = _compute_losses_flona(
                        ema_model.averaged_model, noise_scheduler, batch_obs_images, batch_floorplan_images,
                        distance, actions, goal_pos, curr_pos, curr_ori,
                        accelerator=accelerator,
                    )
                
                if accelerator.is_main_process:
                    data_log = {}
                    for key, value in losses.items():
                        if key in loggers:
                            logger = loggers[key]
                            logger.log_data(value.item())
                            data_log[logger.full_name()] = logger.latest()
                            accelerator.print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                    
                    if use_swanlab and swanlab_log_freq > 0 and i % swanlab_log_freq == 0:
                        swanlab.log(data_log)
            
            if image_log_freq > 0 and i % image_log_freq == 0 and accelerator.is_main_process:
                with torch.no_grad():
                    visualize_diffusion_action_distribution(
                        ema_model.averaged_model, noise_scheduler, batch_obs_images, batch_floorplan_images,
                        batch_viz_obs_images, batch_viz_floorplan_images, actions, distance,
                        goal_pos, curr_pos, curr_ori, goal_pos_local, goal_pos_resized,
                        curr_pos_resized, curr_ori_resized, accelerator=accelerator, type="train",
                        project_folder=project_folder, epoch=epoch, num_images_log=num_images_log,
                        num_samples=30, use_swanlab=use_swanlab,
                    )


def evaluate_flona(
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    # --- 修改: 接收 accelerator, 移除 device ---
    accelerator: Accelerator,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    swanlab_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_swanlab: bool = True,
):
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)
    loggers = {
        "action_loss": Logger("action_loss", "test", window_size=print_log_freq),
        "action_waypts_cos_sim": Logger("action_waypts_cos_sim", "test", window_size=print_log_freq),
        "multi_action_waypts_cos_sim": Logger("multi_action_waypts_cos_sim", "test", window_size=print_log_freq),
        "dist_loss": Logger("dist_loss", "test", window_size=print_log_freq),
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with torch.no_grad():
        with tqdm.tqdm(
            itertools.islice(dataloader, num_batches), total=num_batches, dynamic_ncols=True,
            desc=f"Evaluating for epoch {epoch}", leave=False, disable=not accelerator.is_main_process
        ) as tepoch:
            for i, data in enumerate(tepoch):
                (obs_image, floorplan_image, actions, distance, goal_pos, curr_pos, curr_ori,
                 goal_pos_resized, curr_pos_resized, curr_ori_resized, goal_pos_local, curr_pos_local) = data
                
                obs_images = torch.split(obs_image, 3, dim=1)
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_floorplan_images = TF.resize(floorplan_image, VISUALIZATION_IMAGE_SIZE[::-1])
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1)
                batch_floorplan_images = transform(floorplan_image)
                
                device = accelerator.device
                h0, w0 = obs_images[0].shape[-2:]
                h1, w1 = batch_viz_obs_images.shape[-2:]
                goal_pos_resized = goal_pos_resized * torch.tensor([w1/w0, h1/h0], device=device)
                curr_pos_resized = curr_pos_resized * torch.tensor([w1/w0, h1/h0], device=device)
                curr_ori_resized = curr_ori_resized * torch.tensor([w1/w0, h1/h0], device=device)
                for b in range(goal_pos_resized.shape[0]):
                    batch_viz_floorplan_images[b,:,int(goal_pos_resized[b,1]), int(goal_pos_resized[b,0])] = torch.tensor(RED, device=device)
                    batch_viz_floorplan_images[b,:,int(curr_pos_resized[b,1]), int(curr_pos_resized[b,0])] = torch.tensor(GREEN, device=device)
                    batch_viz_floorplan_images[b,:,int(curr_ori_resized[b,1]), int(curr_ori_resized[b,0])] = torch.tensor(CYAN, device=device)

                losses = _compute_losses_flona(
                    ema_model, noise_scheduler, batch_obs_images, batch_floorplan_images,
                    distance, actions, goal_pos, curr_pos, curr_ori,
                    accelerator=accelerator,
                )
                
                if accelerator.is_main_process:
                    data_log = {}
                    for key, value in losses.items():
                        if key in loggers:
                            logger = loggers[key]
                            logger.log_data(value.item())
                            data_log[logger.full_name()] = logger.latest()
                            if print_log_freq > 0 and i % print_log_freq == 0:
                                accelerator.print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                    
                    if use_swanlab and swanlab_log_freq > 0 and i % swanlab_log_freq == 0:
                        swanlab.log(data_log)
                
                if image_log_freq > 0 and i % image_log_freq == 0 and accelerator.is_main_process:
                    visualize_diffusion_action_distribution(
                        ema_model, noise_scheduler, batch_obs_images, batch_floorplan_images,
                        batch_viz_obs_images, batch_viz_floorplan_images, actions, distance,
                        goal_pos, curr_pos, curr_ori, goal_pos_local, goal_pos_resized,
                        curr_pos_resized, curr_ori_resized, accelerator=accelerator, type="evaluate",
                        project_folder=project_folder, epoch=epoch, num_images_log=num_images_log,
                        num_samples=30, use_swanlab=use_swanlab,
                    )


def execute_model(
    model: nn.Module,
    cur_pos: np.ndarray,
    cur_heading: np.ndarray,
    goal_pos: np.ndarray,
    cur_obs: torch.Tensor,
    floorplan: torch.Tensor,
    metric_waypoint_spacing: float,
    waypoint_spacing: float,
    transform: transforms,
    # --- 修改: 接收 accelerator, 移除 device ---
    accelerator: Accelerator,
    noise_scheduler: DDPMScheduler,
    floorplan_ary: np.ndarray,
    log_add: str = None,
):
    model.eval()
    device = accelerator.device # 获取当前设备

    # 将 numpy array 转换为 tensor 并放到正确的设备上
    cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32, device=device)
    cur_heading = torch.as_tensor(cur_heading, dtype=torch.float32, device=device)
    goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32, device=device)
    cur_obs = torch.as_tensor(cur_obs, dtype=torch.float32, device=device)
    floorplan = torch.as_tensor(floorplan, dtype=torch.float32, device=device)

    cur_pos /= metric_waypoint_spacing * waypoint_spacing
    cur_heading /= metric_waypoint_spacing * waypoint_spacing
    goal_pos /= metric_waypoint_spacing * waypoint_spacing
    
    cur_obss = torch.split(cur_obs, 1, dim=0)
    batch_cur_obss = [transform(obs) for obs in cur_obss]
    batch_cur_obss = torch.cat(batch_cur_obss, dim=1)
    batch_floorplan = transform(floorplan)
    
    with torch.no_grad():
        model_output_dict = model_output(
            model,
            noise_scheduler,
            batch_cur_obss,
            batch_floorplan,
            32,
            2,
            30,
            goal_pos,
            cur_pos,
            cur_heading,
            accelerator=accelerator,
        )
    actions = model_output_dict['actions'].mean(dim=0)
    actions_normed_global = to_global_coords(to_numpy(actions), to_numpy(cur_pos).squeeze(0), to_numpy(cur_heading).squeeze(0))
    actions_meter_global = actions_normed_global * metric_waypoint_spacing * waypoint_spacing
    
    return actions_meter_global


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_floorplan_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    # --- 修改: 接收 accelerator, 移除 device ---
    accelerator: Accelerator,
):
    device = accelerator.device # 获取当前设备

    obsfloorplan_cond_fused = model("vision_encoder", obs_img=batch_obs_images, floorplan_img=batch_floorplan_images, obs_pos=curr_pos, goal_pos=goal_pos, obs_ori=curr_ori)
    obsfloorplan_cond_fused = obsfloorplan_cond_fused.repeat_interleave(num_samples, dim=0)

    noisy_diffusion_output = torch.randn((len(obsfloorplan_cond_fused), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsfloorplan_cond_fused
        )
        diffusion_output = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=diffusion_output).prev_sample

    actions = get_action(diffusion_output, ACTION_STATS)
    distance = model("dist_pred_net", obsgoal_cond=obsfloorplan_cond_fused)

    return {'actions': actions, 'distance': distance}


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_floorplan_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_floorplan_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    batch_curr_pos: torch.Tensor,
    batch_curr_ori: torch.Tensor,
    batch_goal_pos_local: torch.Tensor,
    batch_goal_pos_resized: torch.Tensor,
    batch_curr_pos_resized: torch.Tensor,
    batch_curr_ori_resized: torch.Tensor,
    accelerator: Accelerator,
    type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_swanlab: bool = True,
):
    """Plot samples from the exploration model."""
    visualize_path = os.path.join(project_folder, "visualize", type, f"epoch{epoch}", "action_sampling_prediction")
    os.makedirs(visualize_path, exist_ok=True)

    num_images_log = min(num_images_log, batch_obs_images.shape[0])
    
    # 截取所需数量的数据进行可视化
    obs_imgs = batch_obs_images[:num_images_log]
    floor_imgs = batch_floorplan_images[:num_images_log]
    action_labels = batch_action_label[:num_images_log]
    dist_labels = batch_distance_labels[:num_images_log]
    goal_pos = batch_goal_pos[:num_images_log]
    curr_pos = batch_curr_pos[:num_images_log]
    curr_ori = batch_curr_ori[:num_images_log]
    goal_pos_local = batch_goal_pos_local[:num_images_log]
    goal_pos_resized = batch_goal_pos_resized[:num_images_log]
    curr_pos_resized = batch_curr_pos_resized[:num_images_log]
    curr_ori_resized = batch_curr_ori_resized[:num_images_log]
    viz_obs_imgs = batch_viz_obs_images[:num_images_log]
    viz_floor_imgs = batch_viz_floorplan_images[:num_images_log]

    swanlab_list = []
    pred_horizon = action_labels.shape[1]
    action_dim = action_labels.shape[2]

    model_output_dict = model_output(
        ema_model, noise_scheduler, obs_imgs, floor_imgs,
        pred_horizon, action_dim, num_samples, goal_pos,
        curr_pos, curr_ori, accelerator=accelerator,
    )
    actions_list_np = to_numpy(model_output_dict['actions'])
    distances_list_np = to_numpy(model_output_dict['distance'])

    actions_per_img = np.split(actions_list_np, num_images_log, axis=0)
    distances_per_img = np.split(distances_list_np, num_images_log, axis=0)
    
    dist_labels_np = to_numpy(dist_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        
        pred_actions = actions_per_img[i]
        gt_action = to_numpy(action_labels[i])
        
        traj_list = np.concatenate([pred_actions, gt_action[None]], axis=0)
        traj_colors = ["red"] * len(pred_actions) + ["magenta"]
        traj_alphas = [0.1] * len(pred_actions) + [1.0]

        point_list = [np.array([0, 0]), to_numpy(goal_pos_local[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        # 调用修复后的绘图函数
        plot_trajs_and_points(
            ax=ax[0],
            list_trajs=traj_list,
            list_points=point_list,
            traj_colors=traj_colors,
            point_colors=point_colors,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
            # 注意：不再传递 traj_labels，让函数内部安全处理
        )
        
        obs_image = to_numpy(viz_obs_imgs[i])
        floorplan_image = to_numpy(viz_floor_imgs[i])
        obs_image = np.moveaxis(obs_image, 0, -1)
        floorplan_image = np.moveaxis(floorplan_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(floorplan_image)

        distances_avg = np.mean(distances_per_img[i])
        distances_std = np.std(distances_per_img[i])
        
        ax[0].set_title("Diffusion Action Predictions")
        ax[1].set_title("Observation")
        ax[2].set_title(f"Goal: label={dist_labels_np[i]:.2f} pred={distances_avg:.2f}±{distances_std:.2f}")
        
        # ... (其他绘图设置保持不变)
        fig.set_size_inches(18.5, 10.5)
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        if use_swanlab:
            swanlab_list.append(swanlab.Image(save_path))
        plt.close(fig)
        
    if len(swanlab_list) > 0 and use_swanlab:
        swanlab.log({f"{type}_action_samples": swanlab_list})


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list,
    point_colors: list,
    traj_labels: Optional[list] = None, # <-- 1. 默认值改为 None
    point_labels: Optional[list] = None,
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
):
    """
    一个更稳健的绘图函数，安全处理标签。
    """
    for i, traj in enumerate(list_trajs):
        # 2. 安全地检查标签是否存在且长度足够
        label = traj_labels[i] if traj_labels and i < len(traj_labels) else None
        color = traj_colors[i] if i < len(traj_colors) else 'blue' # 如果颜色不够用蓝色保底
        alpha = traj_alphas[i] if traj_alphas and i < len(traj_alphas) else 1.0

        ax.plot(
            traj[:, 0], 
            traj[:, 1], 
            color=color,
            alpha=alpha,
            marker="o",
            label=label
        )
        
    for i, pt in enumerate(list_points):
        label = point_labels[i] if point_labels and i < len(point_labels) else None
        color = point_colors[i] if i < len(point_colors) else 'black'
        alpha = point_alphas[i] if point_alphas and i < len(point_alphas) else 1.0

        ax.plot(
            pt[0], 
            pt[1], 
            color=color, 
            alpha=alpha,
            marker="o",
            markersize=7.0,
            label=label
        )

    if traj_labels is not None or point_labels is not None:
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def get_data_stats(data):
    # (此函数无需修改，保持原样)
    data = data.reshape(-1,data.shape[-1])
    stats = {'min': np.min(data, axis=0), 'max': np.max(data, axis=0)}
    return stats

def normalize_data(data: torch.Tensor, stats: dict):
    # --- 这是适配 accelerate 的修改 ---
    # 将 stats 从 numpy array 转换为 tensor，并放到与 data 相同的设备上
    stats_min = torch.from_numpy(stats['min']).to(data.device, non_blocking=True)
    stats_max = torch.from_numpy(stats['max']).to(data.device, non_blocking=True)
    
    # 归一化到 [0,1]
    ndata = (data - stats_min) / (stats_max - stats_min)
    # 归一化到 [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata: torch.Tensor, stats: dict):
    # --- 这是适配 accelerate 的修改 ---
    # 将 stats 从 numpy array 转换为 tensor，并放到与 ndata 相同的设备上
    stats_min = torch.from_numpy(stats['min']).to(ndata.device, non_blocking=True)
    stats_max = torch.from_numpy(stats['max']).to(ndata.device, non_blocking=True)

    ndata = (ndata + 1) / 2
    data = ndata * (stats_max - stats_min) + stats_min
    return data

def get_delta(actions: torch.Tensor):
    # --- 这是适配 accelerate 的修改 ---
    # 使用 torch.zeros 和 torch.cat 替代 numpy 函数
    # 这让整个操作都在 GPU 上完成，避免了数据拷贝，效率更高
    zeros = torch.zeros((actions.shape[0], 1, actions.shape[-1]), device=actions.device, dtype=actions.dtype)
    ex_actions = torch.cat([zeros, actions], dim=1)
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta

def get_action(diffusion_output: torch.Tensor, action_stats: dict = ACTION_STATS):
    # --- 这是适配 accelerate 的修改 ---
    # 将整个函数重写为纯 PyTorch 版本，以避免数据类型冲突和不必要的 CPU-GPU 拷贝

    # 1. 重塑形状，输入 diffusion_output 已经是 Tensor
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    
    # 2. 直接调用我们修改过的、期望 Tensor 输入的 unnormalize_data 函数
    deltas = unnormalize_data(ndeltas, action_stats)
    
    # 3. 使用 torch.cumsum 替代 np.cumsum
    actions = torch.cumsum(deltas, dim=1)
    
    # 4. 直接返回 Tensor，不再需要 from_numpy
    return actions

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # (此函数无需修改，保持原样)
    return tensor.detach().cpu().numpy()

def from_numpy(array: np.ndarray) -> torch.Tensor:
    # (此函数无需修改，保持原样)
    return torch.from_numpy(array).float()