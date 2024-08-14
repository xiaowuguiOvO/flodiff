import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from training.Logger import Logger
from model.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from model.data_utils import to_global_coords, img_path_to_data_and_point_transfer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


ACTION_STATS = {}
ACTION_STATS["min"] = np.array([-2.5, -4])
ACTION_STATS["max"] = np.array([5, 4])    
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,  
    curr_ori: torch.Tensor,
    device: torch.device,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]
    num_samples = 1
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples,
        goal_pos,
        curr_pos,
        curr_ori,
        device=device,
    )
    actions = model_output_dict['actions']
    distance = model_output_dict['distance']

    dist_loss = F.mse_loss(distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()


    action_loss = action_reduce(F.mse_loss(actions, batch_action_label, reduction="none"))

    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))

    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "action_loss": action_loss,
        "action_waypts_cos_sim": action_waypts_cos_similairity,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
        "dist_loss": dist_loss,
    }

    return results

def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    num_batches = len(dataloader)

    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "dist_loss": dist_loss_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                curr_pos,
                curr_ori,
                goal_pos_resized,
                curr_pos_resized,
                curr_ori_resized,
                goal_pos_local,
                curr_pos_local
            ) = data

            
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)

            # plot goal and current position on images
            h0, w0 = obs_images[0].shape[-2:]
            h1, w1 = batch_viz_obs_images.shape[-2:]
            goal_pos_resized = goal_pos_resized * torch.tensor([w1/w0, h1/h0])
            curr_pos_resized = curr_pos_resized * torch.tensor([w1/w0, h1/h0])
            curr_ori_resized = curr_ori_resized * torch.tensor([w1/w0, h1/h0])
            for b in range(goal_pos_resized.shape[0]):
                batch_viz_goal_images[b,:,int(goal_pos_resized[b,1]), int(goal_pos_resized[b,0])] = torch.tensor(RED)
                batch_viz_goal_images[b,:,int(curr_pos_resized[b,1]), int(curr_pos_resized[b,0])] = torch.tensor(GREEN)
                batch_viz_goal_images[b,:,int(curr_ori_resized[b,1]), int(curr_ori_resized[b,0])] = torch.tensor(CYAN)

            B = actions.shape[0]

            # Generate random goal mask
            # obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, obs_pos = curr_pos, goal_pos = goal_pos, obs_ori = curr_ori, input_goal_mask=None)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, obs_pos = curr_pos, goal_pos = goal_pos, obs_ori = curr_ori, input_goal_mask=None)
            
            # Get distance label
            distance = distance.float().to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)              
                return unreduced_loss.mean()
            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            wandb.log({"total_loss": loss_cpu})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            goal_pos,
                            curr_pos,
                            curr_ori,
                            device,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    curr_pos,
                    curr_ori,
                    goal_pos_local,
                    goal_pos_resized,
                    curr_pos_resized,
                    curr_ori_resized,
                    
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

def evaluate_nomad(
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    action_loss_logger = Logger("action_loss", "test", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "test", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "test", window_size=print_log_freq
    )
    dist_loss_logger = Logger("dist_loss", "test", window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "dist_loss": dist_loss_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),        # only evaluate on a subset of the data
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,    # (b, predict_len, 2)      local coordinate    waypoints(steps) metric
                distance,   # (b,)    step from current to goal
                goal_pos,   # (b, 2)
                curr_pos,   # (b, 2)
                curr_ori,   # (b, 2)      global coordinate   waypoints metric
                goal_pos_resized, # (b, 2)  goal_pos_resized
                curr_pos_resized, # (b, 2)  curr_pos_resized
                curr_ori_resized,  # (b, 2)  curr_ori_resized
                goal_pos_local, # (b, 2)  goal_pos_local
                curr_pos_local  # (b, 2)  curr_pos_local
            ) = data
            
            obs_images = torch.split(obs_image, 3, dim=1)   # tuple, containing L*(b,3,h,w) tensors
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])   # (height, width)
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)  # (b,3*L,h,w)
            batch_goal_images = transform(goal_image).to(device)  # (b,3,h,w)
            
            # plot goal and current position on images
            h0, w0 = obs_images[0].shape[-2:]
            h1, w1 = batch_viz_obs_images.shape[-2:]
            goal_pos_resized = goal_pos_resized * torch.tensor([w1/w0, h1/h0])
            curr_pos_resized = curr_pos_resized * torch.tensor([w1/w0, h1/h0])
            curr_ori_resized = curr_ori_resized * torch.tensor([w1/w0, h1/h0])
            for b in range(goal_pos_resized.shape[0]):
                batch_viz_goal_images[b,:,int(goal_pos_resized[b,1]), int(goal_pos_resized[b,0])] = torch.tensor(RED)
                batch_viz_goal_images[b,:,int(curr_pos_resized[b,1]), int(curr_pos_resized[b,0])] = torch.tensor(GREEN)
                batch_viz_goal_images[b,:,int(curr_ori_resized[b,1]), int(curr_ori_resized[b,0])] = torch.tensor(CYAN)

            B = actions.shape[0]

            # Generate random goal mask
            # rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            # goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            # no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, obs_pos = curr_pos, goal_pos = goal_pos, obs_ori = curr_ori, input_goal_mask=None)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            

            distance = distance.to(device)

            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)   # seem like direction of action
            assert naction.shape[-1] == 2, "action dim must be 2"

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Logging
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            if use_wandb:
                wandb.log({"diffusion_eval_loss": loss})
            else:
                print("diffusion_eval_loss", loss)

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            goal_pos,
                            curr_pos,
                            curr_ori,
                            device,
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
            
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,

                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    curr_pos,
                    curr_ori,
                    goal_pos_local,
                    goal_pos_resized,
                    curr_pos_resized,
                    curr_ori_resized,
                    
                    device,
                    "evaluate",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )

def execute_model(
    ema_model: EMAModel,
    cur_pos: np.ndarray,        # np.array (1,2)   
    cur_heading: np.ndarray,    # np.array (1,2)
    cur_pos_f3: np.ndarray, # np.array (1,3)      
    cur_heading_f3: np.ndarray, # np.array (1,3)  
    goal_pos: np.ndarray,       # np.array (1,2)
    # img_paths: List[str],       # list of image paths
    # floorplan_path: str,        # floorplan path
    cur_obs: torch.Tensor,        # torch.tensor(L,3,h,w)
    floorplan: torch.Tensor,      # torch.tensor (1,3,h,w)
    metric_waipoint_spacing: float,
    waypoint_spacing: float,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    floorplan_ary: np.ndarray,
    log_add: str = None,
):
    """
    Execute the model on the given data.
    Args:
        ema_model: exponential moving average model
        cur_pos: current position
        goal_pos: goal position
        # img_paths: list of image paths
        # floorplan_path: floorplan path
        cur_obs: current observation
        floorplan: floorplan
        transform: transform to apply to images
        device: device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        log_folder: folder to save images to
    """
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    cur_pos = torch.as_tensor(cur_pos, dtype=torch.float32)
    cur_heading = torch.as_tensor(cur_heading, dtype=torch.float32)
    goal_pos = torch.as_tensor(goal_pos, dtype=torch.float32)
    cur_obs = torch.as_tensor(cur_obs, dtype=torch.float32)
    floorplan = torch.as_tensor(floorplan, dtype=torch.float32)
    cur_pos /= metric_waipoint_spacing * waypoint_spacing
    cur_heading /= metric_waipoint_spacing * waypoint_spacing
    goal_pos /= metric_waipoint_spacing * waypoint_spacing
    cur_obss = torch.split(cur_obs, 1, dim=0)
    batch_cur_obss = [transform(obs) for obs in cur_obss]
    batch_cur_obss = torch.cat(batch_cur_obss, dim=1).to(device)  # (1,3*L,h,w)
    batch_floorplan = transform(floorplan).to(device)  # (1,3,h,w)
    
    cur_pos_f3 = torch.as_tensor(cur_pos_f3, dtype=torch.float32)
    cur_heading_f3 = torch.as_tensor(cur_heading_f3, dtype=torch.float32)
    
    # _, cur_pos_resized, goal_pos_resized, cur_ori_resized = img_path_to_data_and_point_transfer('/home/user/data/vis_nav/iGibson/igibson/dataset/Quantico_220/train/Quantico/traj_127/00072.png', (96, 96), cur_pos[0], goal_pos[0], cur_heading[0])
    # cur_pos_i = torch.tensor(np.array([cur_pos_resized]))
    # goal_pos_i = torch.tensor(np.array([goal_pos_resized]))
    # cur_heading_i = torch.tensor(np.array([cur_ori_resized]))
    
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_cur_obss,
        batch_floorplan,
        32,
        2,
        30,
        goal_pos,
        cur_pos_f3,
        cur_heading_f3,
        device=device,
    )
    actions = model_output_dict['actions'].mean(dim=0)  # [1,8,2]
    
    # actions = actions.squeeze(0)
    distance = model_output_dict['distance']
    # pos_ori = model_output_dict['pos_ori']
    # print('pos_ori shape ', pos_ori.shape)
    # print(pos_ori.mean(dim=0))
    # print('gt ', cur_pos, cur_heading )
    actions_normed_global = to_global_coords(to_numpy(actions), to_numpy(cur_pos).squeeze(0), to_numpy(cur_heading).squeeze(0))
    actions_meter_global = actions_normed_global * metric_waipoint_spacing * waypoint_spacing
    
    
    # if f3
    # actions_meter_global_transformed = actions_meter_global.copy()
    # for i in range(actions_meter_global.shape[0]):
    #     actions_meter_global_transformed[i] = actions_meter_global[i] - actions_meter_global[0] + cur_pos.squeeze(0)
    
    if log_add is not None:
        save_action = actions.cpu().detach().numpy()
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace = 0.9, hspace = 0.7)
        ax1 = plt.subplot(gs[:2, :2])
        ax2 = plt.subplot(gs[:2, 2:])
        ax3 = plt.subplot(gs[2:, :3])
        ax4 = plt.subplot(gs[2:, 3:])
        
        goal_pos_metric = goal_pos * metric_waipoint_spacing * waypoint_spacing
        floor_width = floorplan_ary.shape[0]
        end_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
        start_xy = np.flip((np.array(goal_pos_metric[0]) / 0.01 + floor_width / 2.0)).astype(int)
        floorplan_ary[max(0, end_xy[0]-5) : min(end_xy[0]+5, floorplan_ary.shape[0]), max(0, end_xy[1]-5) : min(end_xy[1]+5, floorplan_ary.shape[1]), :] = np.array([0, 0, 255, 255])
        
        ax1.imshow(cur_obs[-1].permute(1,2,0).cpu().detach().numpy())
        ax2.plot(save_action[:,0], save_action[:,1], marker = '.')
        for i, xy in enumerate(actions_meter_global):
            map_xy = np.flip((np.array(xy) / 0.01 + floor_width / 2.0)).astype(int)
            if i == 0:
                start_xy = map_xy
            if i < 8:
                color = np.array([255, 0, 0, 255])
                floorplan_ary[map_xy[0]-2 : map_xy[0]+2, map_xy[1]-2 : map_xy[1]+2, :] = color
            else:
                color = np.array([0, 255, 0, 30])
                floorplan_ary[map_xy[0]-1 : map_xy[0]+1, map_xy[1]-1 : map_xy[1]+1, :] = color
            
        ax3.imshow(floorplan_ary[max(0, start_xy[0]-200) : min(start_xy[0]+200, floorplan_ary.shape[0]), max(0, start_xy[1]-200) : min(start_xy[1]+200, floorplan_ary.shape[1]), :])
        ax4.imshow(floorplan_ary)
        # plt.plot(actions_meter_global[:,0], actions_meter_global[:,1], marker = 'o')
        plt.savefig(os.path.join(log_add))
    
    return actions_meter_global    
    


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):       # (0,0)->first action point, first action point->second action point, ...
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    goal_pos: torch.Tensor,
    curr_pos: torch.Tensor,
    curr_ori: torch.Tensor,
    device: torch.device,
):

    obsgoal_cond, obsgoal_cond_fused = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, obs_pos=curr_pos, goal_pos=goal_pos, obs_ori = curr_ori, input_goal_mask=None)
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)
    obsgoal_cond_fused = obsgoal_cond_fused.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obsgoal_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond_fused
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    actions = get_action(diffusion_output, ACTION_STATS)
    distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond_fused)
    # pos_ori = model("pos_ori_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'actions': actions,
        'distance': distance,
        # 'pos_ori': pos_ori
    }


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    batch_curr_pos: torch.Tensor,
    batch_curr_ori: torch.Tensor,
    batch_goal_pos_local: torch.Tensor,
    batch_goal_pos_resized: torch.Tensor,
    batch_curr_pos_resized: torch.Tensor,
    batch_curr_ori_resized: torch.Tensor,
    
    device: torch.device,
    type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    batch_curr_pos = batch_curr_pos[:num_images_log]
    batch_curr_ori = batch_curr_ori[:num_images_log]
    batch_goal_pos_local = batch_goal_pos_local[:num_images_log]
    batch_goal_pos_resized = batch_goal_pos_resized[:num_images_log]
    batch_curr_pos_resized = batch_curr_pos_resized[:num_images_log]
    batch_curr_ori_resized = batch_curr_ori_resized[:num_images_log]
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)
    actions_list = []
    distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            batch_goal_pos,
            batch_curr_pos,
            batch_curr_ori,
            device,
        )
        actions_list.append(to_numpy(model_output_dict['actions'])) # local, waypoints metric
        distances_list.append(to_numpy(model_output_dict['distance']))
    
    # for save
    # actions_list_global = []
    # for b in range(len(actions_list[0][0])):
    #     cor_normed_global = to_global_coords(actions_list[0][0][0], batch_curr_pos[0], batch_curr_ori[0])
    #     cor_metric_global = cor_normed_global * 0.255
    #     actions_list_global.append(cor_metric_global)
    # actions_list_global = np.concatenate(actions_list_global, axis=0).reshape(-1,2)
    # # save actions_list_global
    # ddir = actions_list_global[1:]
    # save_arr = np.concatenate([actions_list_global[:-1], ddir], axis=1)
    # save_arr = np.concatenate([np.array([batch_curr_pos[0][0], batch_curr_pos[0][1], batch_curr_ori[0][0], batch_curr_ori[0][1]]).reshape(1,4)*0.255, save_arr], axis=0)
    # np.savetxt(os.path.join(visualize_path, f"actions_list_global_{type}.txt"), save_arr)
    

    # concatenate
    actions_list = np.concatenate(actions_list, axis=0)
    distances_list = np.concatenate(distances_list, axis=0)


    # split into actions per observation
    actions_list = np.split(actions_list, num_images_log, axis=0)
    distances_list = np.split(distances_list, num_images_log, axis=0)


    distances_avg = [np.mean(dist) for dist in distances_list]
    distances_std = [np.std(dist) for dist in distances_list]

    assert len(actions_list) == len(actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        actions = actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            actions,
            action_label[None],
        ], axis=0)
        # print("traj_list.shape", traj_list.shape)   
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(actions) + ["magenta"]
        traj_alphas = [0.1] * len(actions) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        # point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos_local[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={distances_avg[i]:.2f}±{distances_std[i]:.2f}")
        
        str_text = f'goal_resized:{batch_goal_pos_resized[i].cpu().numpy()} curr_pos_resized:{batch_curr_pos_resized[i].cpu().numpy()} curr_ori_resized:{batch_curr_ori_resized[i].cpu().numpy()}'
        fig.text(0, 0, str_text)
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        # wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{type}_action_samples": wandb_list}, commit=False)

def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    Plot trajectories and points that could potentially have a yaw.

    Args:
        ax: matplotlib axis
        list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) (if there is no yaw) or (horizon, 4) (if there is yaw)
        list_points: list of points, each point is a numpy array of shape (2,)
        traj_colors: list of colors for trajectories
        point_colors: list of colors for points
        traj_labels: list of labels for trajectories
        point_labels: list of labels for points
        traj_alphas: list of alphas for trajectories
        point_alphas: list of alphas for points
        quiver_freq: frequency of quiver plot (if the trajectory data includes the yaw of the robot)
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")

