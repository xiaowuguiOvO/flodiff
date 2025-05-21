import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import DDPMScheduler
class FloNaAgent:
    def __init__(self, model_path=None):
        # 初始化模型
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 初始化噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
        
        # 初始化图像转换
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        # 加载训练好的模型
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            # print(f"模型已加载: {model_path}")
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