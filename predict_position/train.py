import tqdm
import torch
import os
import argparse
import yaml
import time
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler


from predict_position import position_dataset
from predict_model import vision_encoder, replace_bn_with_gn
from predict_model import position_predictor, DenseNetwork

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import sys
sys.path.append("../")
from training.train_eval_loop import load_model
from training.Logger import Logger



def train(config):
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")
    
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
        
    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)
    
    if "clip_goals" not in config:
        config["clip_goals"] = False

    data_config = config["datasets"]
    
    # Loda data
    train_dataset = position_dataset.position_dataset(
        data_folder = os.path.join(data_config["data_folder"], "train"),
        scene_name = "Quantico",
        image_size = config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        obs_type=config["obs_type"],
        goal_type=config["goal_type"],
    )
    test_dataset = position_dataset.position_dataset(
        data_folder = os.path.join(data_config["data_folder"], "test"),
        scene_name = "Quantico_t",
        image_size = config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        obs_type=config["obs_type"],
        goal_type=config["goal_type"],
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,)
    
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
        
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    
    

    #create model
    condition_encoder = vision_encoder(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"]
    )
    condition_encoder = replace_bn_with_gn(condition_encoder)

    noise_predictor = ConditionalUnet1D(
        input_dim = 4,
        global_cond_dim = config["encoding_size"],
        down_dims = config["down_dims"],
        cond_predict_scale = config["cond_predict_scale"],
    )
    
    pos_ori_predictor = DenseNetwork(config["encoding_size"])
    
    
    model = position_predictor(condition_encoder, noise_predictor, pos_ori_predictor)  
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps = config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    ) 
    
    
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )
    
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
    
    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    
    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    #train model
    latest_path = os.path.join(config["project_folder"], f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)
    alpha = float(config["alpha"])
    current_epoch = 0
    epochs = config["epochs"]
    use_wandb = config["use_wandb"]
    wandb_log_freq = config["wandb_log_freq"]
    print_log_freq = config["print_log_freq"]
    for epoch in range(current_epoch, current_epoch + epochs):
        print(f"training: epoch_ {epoch}")
        num_batches = len(train_loader)
        model.train()
        
        diff_pos_ori_pred_loss_logger = Logger("diff_pos_ori_pred_loss", "train", window_size=print_log_freq)
        diff_pos_ori_pred_cos_logger = Logger("diff_pos_ori_pred_cos", "train", window_size=print_log_freq)
        diff_pos_pred_loss_logger = Logger("diff_pos_pred_loss", "train", window_size=print_log_freq)
        diff_ori_pred_loss_logger = Logger("diff_ori_pred_loss", "train", window_size=print_log_freq)
        dens_pos_ori_pred_loss_logger = Logger("dens_pos_ori_pred_loss", "train", window_size=print_log_freq)
        dens_pos_ori_pred_cos_logger = Logger("dens_pos_ori_pred_cos", "train", window_size=print_log_freq)
        dens_pos_pred_loss_logger = Logger("dens_pos_pred_loss", "train", window_size=print_log_freq)
        dens_ori_pred_loss_logger = Logger("dens_ori_pred_loss", "train", window_size=print_log_freq)
        loggers = {
            "diff_pos_ori_pred_loss": diff_pos_ori_pred_loss_logger,
            "diff_pos_ori_pred_cos": diff_pos_ori_pred_cos_logger,
            "diff_pos_pred_loss": diff_pos_pred_loss_logger,
            "diff_ori_pred_loss": diff_ori_pred_loss_logger,
            "dens_pos_ori_pred_loss": dens_pos_ori_pred_loss_logger,
            "dens_pos_ori_pred_cos": dens_pos_ori_pred_cos_logger,
            "dens_pos_pred_loss": dens_pos_pred_loss_logger,
            "dens_ori_pred_loss": dens_ori_pred_loss_logger
        }
        with tqdm.tqdm(train_loader, desc = "training batch") as tepoch:
            for i, data in enumerate(tepoch):
                (
                    obs_image,
                    floorplan,
                    position,
                    orientation,
                    position_resized,
                    orientation_resized
                ) = data
                
                obs_images = torch.split(obs_image, 3, dim=1)
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
                batch_floorplan = floorplan.to(device)
                
                B = floorplan.shape[0]
                
                condition = model("condition_encoder", obs=batch_obs_images, floorplan=batch_floorplan)
                
                pos_ori = torch.cat([position, orientation], dim=1).float().to(device)
                pos_ori_pred = model("pos_ori_predictor", condition=condition)
                pos_ori_pred_loss = nn.functional.mse_loss(pos_ori_pred, pos_ori)
                
                # Sample noise to add to actions
                pos_ori_diff = pos_ori.unsqueeze(1).repeat(1, 4, 1)
                noise = torch.randn(pos_ori_diff.shape, device=device)
                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()
                # Add noise to the clean images according to the noise magnitude at each diffusion iteration
                noisy_pos_ori_diff = noise_scheduler.add_noise(pos_ori_diff, noise, timesteps)
                # Predict the noise residual
                noise_pred = model("noise_predictor", sample=noisy_pos_ori_diff, timestep=timesteps, global_cond=condition)
                
                def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                    while unreduced_loss.dim() > 1:
                        unreduced_loss = unreduced_loss.mean(dim=-1)              
                    return unreduced_loss.mean()
                # L2 loss
                diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
                
                # Total loss
                loss = alpha * pos_ori_pred_loss + (1 - alpha) * diffusion_loss
                
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
                wandb.log({"pos_ori_pred_loss": pos_ori_pred_loss.item()})
                wandb.log({"diffusion_loss": diffusion_loss.item()})
                
                # log using ema average
                if i % print_log_freq == 0:
                    pred_horizon = pos_ori_diff.shape[1]
                    pred_dim = pos_ori_diff.shape[2]
                    num_samples = 1
                    log_model = ema_model.averaged_model
                    log_condition = log_model("condition_encoder", obs=batch_obs_images, floorplan=batch_floorplan)
                    log_condition = log_condition.repeat_interleave(num_samples, dim=0)
                    
                    noisy_diffusion_output = torch.randn(
                        (len(log_condition), pred_horizon, pred_dim), device=device)
                    diffusion_output = noisy_diffusion_output
                
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = log_model(
                            "noise_predictor",
                            sample=diffusion_output,
                            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                            global_cond=log_condition
                        )

                        # inverse diffusion step (remove noise)
                        diffusion_output = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=diffusion_output
                        ).prev_sample
                
                    pos_ori_from_diff = diffusion_output.mean(dim=1)
                    pos_ori_from_dens = log_model("pos_ori_predictor", condition=log_condition)
                    # pos_ori_from_diff = pos_ori_from_diff.reshape(pos_ori_from_dens.shape)
                    
                    diff_pos_ori_pred_loss = nn.functional.mse_loss(pos_ori_from_diff.squeeze(1), pos_ori)
                    diff_pos_ori_pred_cos = action_reduce(nn.functional.cosine_similarity(pos_ori_from_diff.squeeze(1), pos_ori, dim=-1))
                    diff_pos_pred_loss = nn.functional.mse_loss(pos_ori_from_diff.squeeze(1)[:, :2], pos_ori[:, :2])
                    diff_ori_pred_loss = nn.functional.mse_loss(pos_ori_from_diff.squeeze(1)[:, 2:], pos_ori[:, 2:])
                    
                    dens_pos_ori_pred_loss = nn.functional.mse_loss(pos_ori_from_dens, pos_ori)
                    dens_pos_ori_pred_cos = action_reduce(nn.functional.cosine_similarity(pos_ori_from_dens, pos_ori, dim=-1))
                    dens_pos_pred_loss = nn.functional.mse_loss(pos_ori_from_dens[:, :2], pos_ori[:, :2])
                    dens_ori_pred_loss = nn.functional.mse_loss(pos_ori_from_dens[:, 2:], pos_ori[:, 2:])
                    
                    log_res = {
                        "diff_pos_ori_pred_loss": diff_pos_ori_pred_loss,
                        "diff_pos_ori_pred_cos": diff_pos_ori_pred_cos,
                        "diff_pos_pred_loss": diff_pos_pred_loss,
                        "diff_ori_pred_loss": diff_ori_pred_loss,
                        "dens_pos_ori_pred_loss": dens_pos_ori_pred_loss,
                        "dens_pos_ori_pred_cos": dens_pos_ori_pred_cos,
                        "dens_pos_pred_loss": dens_pos_pred_loss,
                        "dens_ori_pred_loss": dens_ori_pred_loss
                    }
                    for key, value in log_res.items():
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
        scheduler.step()
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)
    
        project_folder = config["project_folder"]
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(scheduler.state_dict(), latest_scheduler_path)
        
    wandb.log({})

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Position and Orientation Prediction")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="pos_ori_predict_train.yaml",
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
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="ljxjiaxinli-Beijing Institute of Technology", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    train(config)

