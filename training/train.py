import os
import argparse
import yaml
import time

import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.flona_dataset import flona_Dataset
from train_eval_loop import train_eval_loop_flona, load_model


def main(config):
    # ==============================Dataset==============================
    data_config = config["datasets"]
    train_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "train"),
        scene_names=data_config['scene_names'],
        image_size=config["image_size"],
        waypoint_spacing=data_config["waypoint_spacing"],
        len_traj_pred=config["len_traj_pred"],
        context_size=config["context_size"],
        end_slack=data_config["end_slack"],
        normalize=config["normalize"],
    )
    test_dataset = flona_Dataset(
        data_folder=os.path.join(data_config["data_folder"], "test"),
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
            global_cond_dim=config["encoding_size"],    # +6
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])   # +6
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
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
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

    cudnn.benchmark = True  
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = None
  
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

    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    # ==============================Train==============================
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
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        image_log_freq=config["image_log_freq"],
        num_images_log=config["num_images_log"],
        current_epoch=current_epoch,
        alpha=float(config["alpha"]),
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )

    print("Done!!!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
    # project setup
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
        config["project_folder"],  # should error if dir already exists to avoid overwriting and old project
    )
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),          
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    main(config)
