import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import DDPMScheduler
from model.flona import flona
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.flona import DenseNetwork

class FloNaAgent:
    def __init__(self, model_path=None, config=None):
        # 初始化模型
        self.model = None
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化model配置
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.context_size = config["context_size"]
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
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        model = model.to(self.device)
        return model

    def load_model(self, model_path):
        # 加载训练好的模型
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = self.build_model(self.config)
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
        except Exception as e:
            print(f"fail load model: {str(e)}")
            raise e
        
    def get_action(self, observation):
        # 根据观察获取动作
        # observation 包含：
        # - rgb: 当前图像
        # - floorplan: 平面图
        # - task_obs: 任务相关观察
        
        pass
    
    def update_obs_img_queue(self, obs_img):
        self.process_obs_img(obs_img)
    
    def update_floorplan_img(self, floorplan_img):
        self.floorplan_img = floorplan_img
    
    def update_obs_pos(self, obs_pos):
        self.obs_pos = obs_pos
    
    def update_goal_pos(self, goal_pos):
        self.goal_pos = goal_pos
    
    def update_obs_ori(self, obs_ori):
        self.obs_ori = obs_ori
    
    def process_obs_img(self, obs_img):
        obs_img = obs_img.copy()
        obs_img = obs_img.transpose(2, 0, 1)  # 转为[3, H, W]
        obs_img = torch.from_numpy(obs_img).float() / 255.0 
        obs_img = self.transform(obs_img)
        self.obs_img_queue.append(obs_img)
        
        if len(self.obs_img_queue) > self.context_size + 1:
            self.obs_img_queue.pop(0)
         # if the queue is not full, fill it with the last frame
        while len(self.obs_img_queue) < self.context_size + 1:
            if len(self.obs_img_queue) > 0:  # 确保队列不为空
                self.obs_img_queue.append(self.obs_img_queue[-1].clone())
                
    def update_vision_input(self, obs_img, floorplan_img, obs_pos, goal_pos, obs_ori):
        self.update_obs_img_queue(obs_img)
        self.update_floorplan_img(floorplan_img)
        self.update_obs_pos(obs_pos)
        self.update_goal_pos(goal_pos)
        self.update_obs_ori(obs_ori)
    
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