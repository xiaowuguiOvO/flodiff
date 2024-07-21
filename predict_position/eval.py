import tqdm
import torch
import os
import argparse
import yaml
import time
import numpy as np
import wandb
from PIL import Image

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
from model.data_utils import img_path_to_data, get_data_path, to_local_coords, resize_and_aspect_crop

def eval(config):
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
    
    checkpoint_dir = '/home/user/data/vis_nav/flona/predict_position/logs/pos_ori_predict/pos_ori_predict_2024_07_08_00_07_32'
    load_model_path = os.path.join(checkpoint_dir, "latest.pth")
    load_ema_path = os.path.join(checkpoint_dir, "ema_260.pth")
    latest_model_checkpoint = torch.load(load_model_path)
    latest_ema_checkpoint = torch.load(load_ema_path)
    load_model(model, latest_model_checkpoint)
    
    model = model.to(device)
    ema_model = EMAModel(model, power=0.75)
    load_model(ema_model.averaged_model, latest_ema_checkpoint)
    model.eval()
    
    
    metric_waypoint_spacing = 0.045
    waypoint_spacing = 1
    img_folder = '/home/user/data/vis_nav/iGibson/igibson/dataset/Quantico_500/test/Quantico_t'
    traj_seq = "traj_492"
    curr_time = 5
    traj_path = os.path.join(img_folder, traj_seq, traj_seq+'.npy')
    context = 3
    floorplan_path = '/home/user/data/vis_nav/iGibson/igibson/dataset/Quantico_500/test/Quantico_t/floorplan.png'
    traj_file = open(traj_path, 'rb')
    traj = np.load(traj_file)
    gt = traj[curr_time + 1]
    gt[2:] = gt[:2] + (gt[2:] - gt[:2]) / np.linalg.norm(gt[2:] - gt[:2])
    imgs = [Image.open(os.path.join(img_folder, traj_seq, "%05d.png"%(curr_time + i))) for i in range(-context, 1)]
    resized_imgs = [resize_and_aspect_crop(img, (96,96), 4 / 3) for img in imgs]
    resized_imgs = torch.cat([transform(img) for img in resized_imgs], dim=0)
    resized_imgs = resized_imgs.unsqueeze(0).to(device)
    floorplan = Image.open(floorplan_path)
    floorplan = resize_and_aspect_crop(floorplan, (96,96), 4 / 3)
    floorplan = transform(floorplan)
    floorplan = floorplan.unsqueeze(0).to(device)
    
    condition = model("condition_encoder", obs=resized_imgs, floorplan=floorplan)
    
    pos_ori_pred = model("pos_ori_predictor", condition=condition)
    pos_ori_pred *= metric_waypoint_spacing * waypoint_spacing
    
    num_samples = 30
    pred_horizon = 4
    action_dim = 4
    diff_cond = condition.repeat_interleave(num_samples, dim=0)
    noisy_diffusion_output = torch.randn((len(diff_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output
    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_predictor",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=diff_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    pos_ori_diff = diffusion_output
    pos_ori_diff = pos_ori_diff.mean(dim=0)
    pos_ori_diff = pos_ori_diff.mean(dim=0)
    pos_ori_diff *= metric_waypoint_spacing * waypoint_spacing
    
    print(pos_ori_pred.squeeze(0).detach().to('cpu').numpy())
    print(pos_ori_diff.detach().to('cpu').numpy())
    print(gt)
    
    
    
    
    
    
    
    



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


    # config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    # config["project_folder"] = os.path.join(
    #     "logs", config["project_name"], config["run_name"]
    # )
    # os.makedirs(
    #     config[
    #         "project_folder"
    #     ],  # should error if dir already exists to avoid overwriting and old project
    # )

    # if config["use_wandb"]:
    #     wandb.login()
    #     wandb.init(
    #         project=config["project_name"],
    #         settings=wandb.Settings(start_method="fork"),
    #         entity="ljxjiaxinli-Beijing Institute of Technology", # TODO: change this to your wandb entity
    #     )
    #     wandb.save(args.config, policy="now")  # save the config file
    #     wandb.run.name = config["run_name"]
    #     # update the wandb args with the training configurations
    #     if wandb.run:
    #         wandb.config.update(config)

    # print(config)
    eval(config)