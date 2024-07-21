import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from model.data_utils import img_path_to_data, get_data_path, to_local_coords

"""
IMPORT YOUR MODEL HERE
"""
from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from model.flona_dataset import flona_Dataset
from training.train_eval_loop import train_eval_loop_nomad, load_model

from training.train_utils import train_nomad, evaluate_nomad, execute_model

def navigate(cur_pos, cur_heading, goal_pos, cur_obs, floorplan, metric_waipoint_spacing, waypoint_spacing, config):
    """
    Args:
        cur_pos: current position of the agent     np.array (1,2)
        goal_pos: goal position of the agent       np.array (1,2)
        # img_paths: paths of the images              list of str
        # floorplan_path: path of the floorplan image   str
        cur_obs: current observation of the agent  torch.tensor (L,3,h,w)
        floorplan: floorplan of the environment    torch.tensor (1,3,h,w)
    Return:
        action: action of the agent               np.array (X,4) (pos[0], pos[1], head_point[0], head_point[1])
    """
    
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

    # Load the data
    # train_dataset = []
    # test_dataloaders = {}


    if "clip_goals" not in config:
        config["clip_goals"] = False

    data_config = config["datasets"]

    # test_dataset = flona_Dataset(
    #     data_folder=os.path.join(data_config["data_folder"], "test"),
    #     scene_name="Quantico_t",
    #     image_size=config["image_size"],
    #     waypoint_spacing=data_config["waypoint_spacing"],
    #     len_traj_pred=config["len_traj_pred"],
    #     learn_angle=config["learn_angle"],
    #     context_size=config["context_size"],
    #     end_slack=data_config["end_slack"],
    #     goals_per_obs=data_config["goals_per_obs"],
    #     normalize=config["normalize"],
    #     obs_type=config["obs_type"],
    #     goal_type=config["goal_type"],
    # )
    # for data_split_type in ["train", "test"]:
    #     if data_split_type in data_config:
    #             dataset = ViNT_Dataset(
    #                 data_folder=data_config["data_folder"],
    #                 data_split_folder=data_config[data_split_type],
    #                 dataset_name=dataset_name,
    #                 image_size=config["image_size"],
    #                 waypoint_spacing=data_config["waypoint_spacing"],
    #                 min_dist_cat=config["distance"]["min_dist_cat"],
    #                 max_dist_cat=config["distance"]["max_dist_cat"],
    #                 min_action_distance=config["action"]["min_dist_cat"],
    #                 max_action_distance=config["action"]["max_dist_cat"],
    #                 negative_mining=data_config["negative_mining"],
    #                 len_traj_pred=config["len_traj_pred"],
    #                 learn_angle=config["learn_angle"],
    #                 context_size=config["context_size"],
    #                 context_type=config["context_type"],
    #                 end_slack=data_config["end_slack"],
    #                 goals_per_obs=data_config["goals_per_obs"],
    #                 normalize=config["normalize"],
    #                 goal_type=config["goal_type"],
    #             )
    #             if data_split_type == "train":
    #                 train_dataset.append(dataset)
    #             else:
    #                 dataset_type = f"{dataset_name}_{data_split_type}"
    #                 if dataset_type not in test_dataloaders:
    #                     test_dataloaders[dataset_type] = {}
    #                 test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots


    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]


    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=config["eval_batch_size"],
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=False,
    # )

    # Create the model

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
    
    
    
    # load parameters
    flona_dir = '/home/user/data/vis_nav/flona'
    checkpoint_dir = '/home/user/data/vis_nav/flona/training/logs'
    load_flona_path = os.path.join(checkpoint_dir, "flona", "flona_2024_07_02_20_45_50", "latest.pth")
    load_ema_path = os.path.join(checkpoint_dir, "flona", "flona_2024_07_02_20_45_50", "ema_99.pth")
    latest_flona_checkpoint = torch.load(load_flona_path)
    latest_ema_checkpoint = torch.load(load_ema_path)
    load_model(model, latest_flona_checkpoint)
    

    # project_folder = config["project_folder"]
    print_log_freq = 1
    wandb_log_freq = 1
    image_log_freq = 1
    num_images_log = 8
    current_epoch = 0
    eval_fraction = 0.25

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)
    
    ema_model = EMAModel(model=model,power=0.75)
    load_model(ema_model.averaged_model, latest_ema_checkpoint)
    
    model.eval()
    
    
    
    return execute_model(ema_model, cur_pos, cur_heading, goal_pos, cur_obs, floorplan, metric_waipoint_spacing, waypoint_spacing, transform, device, noise_scheduler, os.path.join(flona_dir, 'scripts', 'logs'))

    


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="/home/user/data/vis_nav/flona/training/flona.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()



    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    # config["run_name"] += "_" + 'eval_2024_06_28_14_04_56'
    # config["project_folder"] = os.path.join(
    #     "logs", config["project_name"], config["run_name"]
    # )
    # os.makedirs(
    #     config[
    #         "project_folder"
    #     ],  # should error if dir already exists to avoid overwriting and old project
    # )


    print(config)
    
    cur_pos = np.array([[-1.94, 1.18]])
    cur_heading = np.array([[-1.7, 1]])
    goal_pos = np.array([[2.16, 2.14]])
    img_paths_dir = '/home/user/data/vis_nav/iGibson/igibson/dataset/test/Quantico_t/traj_90'
    cur_time = 4
    context_size = 3
    metric_waipoint_spacing = 0.04
    waypoint_spacing = 1
    img_paths = [os.path.join(img_paths_dir, '%05d.png' % i) for i in range(cur_time - context_size * waypoint_spacing, cur_time + 1)]
    floorplan_path = '/home/user/data/vis_nav/iGibson/igibson/dataset/test/Quantico_t/floorplan.png'
    
    cur_obs = torch.cat([img_path_to_data(img_path, (96,96)) for img_path in img_paths], dim=0)
    cur_obs = cur_obs.unsqueeze(0)
    cur_obs = torch.split(cur_obs, 3, dim=1)
    cur_obs = torch.cat(cur_obs, dim=0)
    floorplan = img_path_to_data(floorplan_path, (96,96))
    floorplan = floorplan.unsqueeze(0)
    
    actions = navigate(cur_pos, cur_heading, goal_pos, cur_obs, floorplan, metric_waipoint_spacing, waypoint_spacing, config)
    print(actions)
    
    
    
    
    
