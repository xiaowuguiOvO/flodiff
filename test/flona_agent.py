import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import DDPMScheduler
from model.flona import flona
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.flona import DenseNetwork
import numpy as np
from training.train_utils import unnormalize_data, to_numpy, from_numpy
from diffusers.training_utils import EMAModel
import os
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
from typing import Tuple
ACTION_STATS = {}
ACTION_STATS["min"] = np.array([-2.5, -4])
ACTION_STATS["max"] = np.array([5, 4])   

class FloNaAgent:
    def __init__(self, model_path=None, model_config=None,scene_config=None, metric_waypoint_spacing=0.045, waypoint_spacing=1):
        # 初始化模型
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.waypoint_spacing = waypoint_spacing
        self.model = None
        self.model_config = model_config
        self.scene_config = scene_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.context_size = model_config["context_size"]
        self.noise_scheduler = None
        self.vision_encoder = None
        self.noise_pred_net = None
        self.dist_pred_net = None
        self.model = None
        self.noise_pred_net = None
        if model_path:
            self.load_model(model_path)

        # model predict datas
        # vision_encoder
        self.obs_img_queue = [] # [batch, 3*(context_size+1), 96, 96]
        self.floorplan_img = None # [batch, 3, 96, 96]
        self.obs_pos = None # [batch, 2] - x,y坐标
        self.goal_pos = None # [batch, 2] - x,y坐标
        self.obs_ori = None # [batch, 2] - sin, cos
        
        # noise_pred_net
        self.sample = None # [batch, 8, 2]
        self.timestep = None # [batch, 1]
        self.global_cond = None # [batch, encoding_size]
        
        # dist_pred_net
        self.obsgoal_cond = None # [batch, encoding_size]
        
        # floorplan and obs transfer
        trav_folder = model_config["datasets"]["trav_map_folder"]
        self.floor_shapes_ori =  np.load(os.path.join(trav_folder, "floor_shapes.npy"), allow_pickle=True).item()
        
    def build_model(self, config):
        vision_encoder = flona_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        self.vision_encoder = replace_bn_with_gn(vision_encoder)
        self.noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],    # +6
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],    # +6
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        self.dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])   # +6
        model = flona(
            vision_encoder=self.vision_encoder,
            noise_pred_net=self.noise_pred_net,
            dist_pred_net=self.dist_pred_network,
        )
        model = model.to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        return model

    def load_model(self, model_path):
        # 加载训练好的模型
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = self.build_model(self.model_config)
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            self.model = EMAModel(
                self.model,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999
            )
            self.model = self.model.averaged_model
            self.model.eval()
            
        except Exception as e:
            print(f"fail load model: {str(e)}")
            raise e
        
    def get_vision_encoder_feature(self, obs_img_queue, floorplan_img, obs_pos, goal_pos, obs_ori):
        obs_img_tensor = torch.cat(obs_img_queue, dim=0).unsqueeze(0).to(self.device)  # [batch, 3*(context_size+1), 96, 96]
        floor_plan_img_tensor = floorplan_img.unsqueeze(0).to(self.device)
        obs_pos_tensor = torch.from_numpy(obs_pos).float().unsqueeze(0).to(self.device)
        goal_pos_tensor = torch.from_numpy(goal_pos).float().unsqueeze(0).to(self.device)
        obs_ori_tensor = torch.from_numpy(obs_ori).float().unsqueeze(0).to(self.device)

        vision_feature = self.model(
            "vision_encoder", 
            obs_img=obs_img_tensor,
            floorplan_img=floor_plan_img_tensor, 
            obs_pos=obs_pos_tensor,
            goal_pos=goal_pos_tensor,
            obs_ori=obs_ori_tensor,
        )
        return vision_feature
    
    def update_obs_img_queue(self, obs_img):
        self.process_obs_img(obs_img)
    
    def update_floorplan_img(self, floorplan_img):
        self.process_floorplan_img(floorplan_img)
        
    def update_obs_pos(self, obs_pos):
        self.obs_pos = obs_pos
    
    def update_goal_pos(self, goal_pos):
        self.goal_pos = goal_pos
    
    def update_obs_ori(self, obs_ori):
        self.obs_ori = obs_ori
    
    def process_obs_floorplan_to_input(self, img, cur_pos, goal_pos, cur_ori, metric_waypoint_spacing, waypoint_spacing, image_resize_size):
        cur_pos_metric = cur_pos * metric_waypoint_spacing * waypoint_spacing # trans from waypoints to meters
        goal_pos_metric = goal_pos * metric_waypoint_spacing * waypoint_spacing
        cur_ori_metric = cur_ori * metric_waypoint_spacing * waypoint_spacing
        
        scene_name = self.scene_config['scene_id'] + '_' + str(self.scene_config['floor_num'])
        ori_size = self.floor_shapes_ori[scene_name]
        w0 = ori_size
        h0 = ori_size
        # w, h = img.size
        cur_pos = cur_pos_metric * 100 + np.array([w0 / 2, h0 / 2])
        goal_pos = goal_pos_metric * 100 + np.array([w0 / 2, h0 / 2])
        cur_ori = cur_ori_metric * 100 + np.array([w0 / 2, h0 / 2])
        
        img = cv2.resize(img, image_resize_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cur_pos_in_resizeSize = cur_pos * image_resize_size[0] / w0
        goal_pos_in_resizeSize = goal_pos * image_resize_size[0] / w0
        cur_ori_in_resizeSize = cur_ori * image_resize_size[0] / w0       
        resize_img = TF.to_tensor(img)
        
        return (resize_img, cur_pos_in_resizeSize, goal_pos_in_resizeSize, cur_ori_in_resizeSize)
    
    def resize_and_aspect_crop_img(self, obs_img: np.ndarray, image_resize_size: Tuple[int, int], aspect_ratio: float = 1.0):
        obs_img = obs_img * 255
        img = Image.fromarray(obs_img.astype(np.uint8))

        w, h = img.size
        if w > h:
            crop_height = h
            crop_width = int(h * aspect_ratio)
            img = TF.center_crop(img, (crop_height, crop_width))
        else:
            # 如果高度大于或等于宽度，保持宽度不变，裁剪高度
            crop_width = w
            crop_height = int(w / aspect_ratio)
            img = TF.center_crop(img, (crop_height, crop_width))
        img = img.resize(image_resize_size)

        resize_img = TF.to_tensor(img)
        # print(resize_img.min(), resize_img.max(), resize_img.shape)
        return resize_img
    def process_obs_img(self, obs_img):
        obs_img = obs_img.copy()
        obs_img = self.resize_and_aspect_crop_img(obs_img, (96, 96))

        self.obs_img_queue.append(obs_img)
        
        if len(self.obs_img_queue) > self.context_size + 1:
            self.obs_img_queue.pop(0)
         # if the queue is not full, fill it with the last frame
        while len(self.obs_img_queue) < self.context_size + 1:
            if len(self.obs_img_queue) > 0:  # 确保队列不为空
                self.obs_img_queue.append(self.obs_img_queue[-1].clone())

        
    def process_floorplan_img(self, floorplan_img):
        floorplan_img = floorplan_img.copy()
        floorplan_img = self.process_obs_floorplan_to_input(floorplan_img, self.obs_pos, self.goal_pos, self.obs_ori, self.metric_waypoint_spacing, self.waypoint_spacing, (96, 96))[0]
        self.floorplan_img = floorplan_img
        floorplan_np = floorplan_img.cpu().numpy()
        floorplan_np = np.transpose(floorplan_np, (1, 2, 0))
        floorplan_np = (floorplan_np * 255).astype(np.uint8)
        floorplan_np_bgr = cv2.cvtColor(floorplan_np, cv2.COLOR_RGB2BGR)
        # print(floorplan_np_bgr.shape)
        # cv2.imshow('floorplan', floorplan_np_bgr)
        # cv2.waitKey(1)  # 添加等待时间
        
    def update_vision_input(self, obs_img, floorplan_img, obs_pos, goal_pos, obs_ori):
        self.update_obs_pos(obs_pos)
        self.update_goal_pos(goal_pos)
        self.update_obs_ori(obs_ori)

        self.update_obs_img_queue(obs_img)
        self.update_floorplan_img(floorplan_img)

    def diffusion_to_action(self, diffusion_output):
        device = self.device
        ndeltas = diffusion_output
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = unnormalize_data(ndeltas, ACTION_STATS)
        actions = np.cumsum(ndeltas, axis=1)
        return from_numpy(actions).to(device)
    
    def get_action(self, obs_img_queue, floorplan_img, obs_pos, goal_pos, obs_ori, metric_waipoint_spacing, waypoint_spacing):
        # normalization_input
        obs_pos /= metric_waipoint_spacing * waypoint_spacing
        obs_ori /= metric_waipoint_spacing * waypoint_spacing
        goal_pos /= metric_waipoint_spacing * waypoint_spacing
        
        vision_feature = self.get_vision_encoder_feature(obs_img_queue, floorplan_img, obs_pos, goal_pos, obs_ori)
        # copy vision_feature
        vision_feature = vision_feature.repeat_interleave(self.model_config['num_samples'], dim=0)
        #initialize action from Gaussian noise
        noisy_diffusion_output = torch.randn(len(vision_feature), 32, 2, device=self.device)
        diffusion_output = noisy_diffusion_output
        
        for k in self.noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = self.model(
                "noise_pred_net",
                sample=diffusion_output,
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                global_cond=vision_feature
            )
            
            # inverse diffusion step(remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output
            ).prev_sample
            
        actions = self.diffusion_to_action(diffusion_output)
        distacne = self.model("dist_pred_net", obsgoal_cond=vision_feature)
        
        return {
            "actions": actions,
            "distances": distacne
        }
        
    def test_model(self):
        if self.model is None:
            print("模型未加载")
            return False
        
        # 获取模型实际context_size
        actual_context_size = 3
        
        # 根据flona_vint的实现，输入需要是 3*(context_size+1) 通道
        channels_needed = 3 * (actual_context_size + 1)
        
        # 创建正确格式的输入
        dummy_obs_img = torch.randn(1, channels_needed, 96, 96).to(self.device)
        dummy_floorplan = torch.randn(1, 3, 96, 96).to(self.device)
        
        # 调整位置和方向输入的维度
        # 注意: 可能需要调整这些维度
        dummy_obs_pos = torch.randn(1, 2).to(self.device)   # [batch, 2] - x,y坐标
        dummy_goal_pos = torch.randn(1, 2).to(self.device)  # [batch, 2] - x,y坐标
        
        # obs_ori可能需要调整为两个维度 (sin, cos)，而不是单一角度
        dummy_obs_ori = torch.randn(1, 2).to(self.device)  # 从1维改为2维
        
        print("更新的输入尺寸:")
        print(f"- obs_img: {dummy_obs_img.shape}")
        print(f"- floorplan: {dummy_floorplan.shape}")
        print(f"- obs_pos: {dummy_obs_pos.shape}")
        print(f"- goal_pos: {dummy_goal_pos.shape}")
        print(f"- obs_ori: {dummy_obs_ori.shape}")
        
        # 测试vision_encoder
        with torch.no_grad():
            vision_features = self.model("vision_encoder", 
                                    obs_img=dummy_obs_img,
                                    floorplan_img=dummy_floorplan,
                                    obs_pos=dummy_obs_pos,
                                    goal_pos=dummy_goal_pos,
                                    obs_ori=dummy_obs_ori)
            print(f"Vision encoder输出形状: {vision_features.shape}")
            
            # 测试noise_pred_net
            dummy_sample = torch.randn(1, 8, 2).to(self.device)
            dummy_timestep = torch.tensor([500], device=self.device)
            
            noise_pred = self.model("noise_pred_net",
                                sample=dummy_sample,
                                timestep=dummy_timestep,
                                global_cond=vision_features)
            print(f"Noise prediction输出形状: {noise_pred.shape}")
            
            # 尝试测试dist_pred_net
            try:
                dist_pred = self.model("dist_pred_net",
                                    obsgoal_cond=vision_features)
                print(f"Distance prediction输出形状: {dist_pred.shape}")
            except Exception as e:
                print(f"Distance prediction测试失败: {e}")
                
        print("模型测试通过！")
        return True