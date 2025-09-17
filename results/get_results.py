import os
import numpy as np
import re
from training.train_utils import *
from model.flona import *
from model.flona_vint import *
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.data_utils import *
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import hsv_to_rgb

def get_last_shortest_path_index(traj_dir: str) -> int:
    """
    获取某个 traj 文件夹中 shortest_paths 目录下最后一个 shortest_path_for_xxxxx.png 的编号
    :param traj_dir: 例如 "datasets/scenes_117/test/Eudora_0/traj_178"
    :return: 最大编号 (int)，如果没有找到则返回 -1
    """
    shortest_dir = os.path.join(traj_dir, "shortest_paths")
    max_id = -1
    if os.path.isdir(shortest_dir):
        for fname in os.listdir(shortest_dir):
            match = re.match(r"shortest_path_for_(\d+)\.png", fname)
            if match:
                num = int(match.group(1))
                if num > max_id:
                    max_id = num
    return max_id

def get_input_data(cur_pos_meter, goal_pos_meter, cur_ori, img_path, floorplan_path, trajectory_name, scene_name, curr_time):
    metric_waypoint_spacing = config["metric_waypoint_spacing"]
    waypoint_spacing = config["waypoint_spacing"]
    context_size = config["context_size"]
    
    if config['normalize']:
        goal_pos_metric = goal_pos_meter / (metric_waypoint_spacing * waypoint_spacing)
        cur_pos_metric = cur_pos_meter / (metric_waypoint_spacing * waypoint_spacing)
        cur_ori /= metric_waypoint_spacing * waypoint_spacing

    context = []
    context_times = list(
    range(
        curr_time + (-context_size * waypoint_spacing),
        curr_time + 1,
        waypoint_spacing,
    )
    )
    context = [(trajectory_name, t) for t in context_times]

    floorplan_image, cur_pos_resized, goal_pos_resized, cur_ori_resized = load_image_and_transform_points(scene_name, trajectory_name, cur_pos_metric, goal_pos_metric, cur_ori, "floorplan")

    # 1. 先生成一个包含所有独立图片张量的列表
    image_list = [load_image(scene_name, f, t) for f, t in context]
    
    # 2. 使用 torch.stack 在第 0 维创建一个新的批次维度 (L)
    obs_image = torch.stack(image_list, dim=0)
    return  cur_ori, obs_image, floorplan_image, cur_pos_resized, goal_pos_resized, cur_ori_resized

def load_image(scene_name, trajectory_name, name): 
    if name == "floorplan":
        image_path = get_data_path(data_folder, scene_name, name)
    else:
        image_path = get_data_path(os.path.join(data_folder, scene_name), trajectory_name, name)
    
    try:   # directedly load from disk
        with open(image_path, "rb") as f:
            result = img_path_to_data(f, image_size)

        return result
        
    except TypeError:
        print(f"Failed to load image {image_path}")
        
def load_image_and_transform_points(scene_name, trajectory_name, cur_pos, goal_pos, cur_ori, name):
    cur_pos_metric = cur_pos * metric_waypoint_spacing * waypoint_spacing # trans from waypoints to meters
    goal_pos_metric = goal_pos * metric_waypoint_spacing * waypoint_spacing
    cur_ori_metric = cur_ori * metric_waypoint_spacing * waypoint_spacing
    
    
    if name == "floorplan":
        image_path = get_data_path(data_folder, scene_name, name)
    else:
        image_path = get_data_path(os.path.join(data_folder, scene_name), trajectory_name, name)

    try:
        with open(image_path, "rb") as f:
            result = img_path_to_data_and_point_transfer(f, floor_shapes_ori[scene_name], image_size, cur_pos_metric, goal_pos_metric, cur_ori_metric)

        return result
    except TypeError:
        print(f"Failed to load image {image_path}")
        
def visualize_robot_inference_with_coords(cur_pos, goal_pos, cur_ori, cur_pos_resized, 
                                         goal_pos_resized, cur_ori_resized, floorplan_image, 
                                         obs_image, predicted_action, trajectory_name, time_step, 
                                         to_global_coords_func, save_path=None, show_obs=True, cur_shortest_path=None):
    """
    可视化机器人推理过程（使用你的坐标转换函数）
    
    Args:
        cur_pos: 当前位置 (全局坐标)
        goal_pos: 目标位置 (全局坐标)
        cur_ori: 当前朝向 (全局坐标)
        cur_pos_resized: 当前位置 (图像坐标)
        goal_pos_resized: 目标位置 (图像坐标)
        cur_ori_resized: 当前朝向 (图像坐标)
        floorplan_image: 地图图像
        obs_image: 观测图像
        predicted_action: 预测的动作 (局部坐标)
        trajectory_name: 轨迹名称
        time_step: 时间步
        to_global_coords_func: 你的坐标转换函数
        save_path: 保存路径
        show_obs: 是否显示观测图像
    """
    
    # 创建子图
    if show_obs and obs_image is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Robot Inference Visualization - {trajectory_name} - Step {time_step}', fontsize=16)
        main_ax = axes[0, 0]
    else:
        fig, main_ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f'Robot Inference Visualization - {trajectory_name} - Step {time_step}', fontsize=16)
    
    # 处理地图图像
    if isinstance(floorplan_image, torch.Tensor):
        if floorplan_image.dim() == 4:  # [B, C, H, W]
            floorplan_np = floorplan_image[0].permute(1, 2, 0).cpu().numpy()
        elif floorplan_image.dim() == 3:  # [C, H, W]
            floorplan_np = floorplan_image.permute(1, 2, 0).cpu().numpy()
        else:
            floorplan_np = floorplan_image.cpu().numpy()
    else:
        floorplan_np = floorplan_image
    
    # 归一化图像到 [0, 1]
    if floorplan_np.max() > 1.0:
        floorplan_np = floorplan_np / 255.0
    
    main_ax.imshow(floorplan_np)
    
    # 使用已经转换好的图像坐标
    cur_pos_img = cur_pos_resized
    goal_pos_img = goal_pos_resized
    
    # 绘制当前位置（绿色圆圈）
    current_circle = Circle(cur_pos_img, radius=2, color='green', alpha=0.8)
    main_ax.add_patch(current_circle)
    # main_ax.text(cur_pos_img[0], cur_pos_img[1]-15, 'Current', 
    #             ha='center', va='top', color='black', fontweight='bold', fontsize=12)
    
    # 绘制目标位置（红色圆圈）
    goal_circle = Circle(goal_pos_img, radius=2, color='red', alpha=0.8)
    main_ax.add_patch(goal_circle)
    # main_ax.text(goal_pos_img[0], goal_pos_img[1]-15, 'Goal',     
    #             ha='center', va='top', color='black', fontweight='bold', fontsize=12)
    
    # 绘制机器人朝向（箭头）
    arrow_length = 10
    ori_angle = np.arctan2(cur_ori[1] - cur_pos[1], cur_ori[0] - cur_pos[0])
    # if isinstance(cur_ori, np.ndarray) and cur_ori.size > 0:
    #     if len(cur_ori) == 2:
    #         # 如果朝向是向量形式，计算角度
    #         ori_angle = np.arctan2(cur_ori[1] - cur_pos[1], cur_ori[0] - cur_pos[0])
    #     else:
    #         ori_angle = cur_ori[0] if cur_ori.ndim > 0 else cur_ori
    # else:
    #     ori_angle = cur_ori
    
    arrow_end_x = cur_pos_img[0] + arrow_length * np.cos(ori_angle)
    arrow_end_y = cur_pos_img[1] + arrow_length * np.sin(ori_angle)
    
    orientation_arrow = FancyArrowPatch(
        cur_pos_img, (arrow_end_x, arrow_end_y),
        arrowstyle='->', mutation_scale=15, color='blue', linewidth=2
    )
    main_ax.add_patch(orientation_arrow)
    
    # 绘制预测轨迹
    if predicted_action is not None:
        if isinstance(predicted_action, torch.Tensor):
            action_np = predicted_action.cpu().numpy()
        else:
            action_np = np.array(predicted_action)


        trajectory_img_coords  = transform_trajectory_to_image_coords(action_np, floor_shapes_ori[scene_name], image_size)
        shortest_path_img_coords = transform_trajectory_to_image_coords(cur_shortest_path, floor_shapes_ori[scene_name], image_size)

        
        # 绘制轨迹点
        for i, point in enumerate(trajectory_img_coords):
            point_circle = Circle(point, radius=0.1, color='cyan', alpha=0.8)
            main_ax.add_patch(point_circle)
        # cur shortest_path
        for i, point in enumerate(shortest_path_img_coords):
            point_circle = Circle(point, radius=0.1, color='red', alpha=0.8)
            main_ax.add_patch(point_circle)

                
    # 添加坐标信息文本
    info_text = f"Global: pos({cur_pos[0]:.2f}, {cur_pos[1]:.2f})\n"
    info_text += f"Image: pos({cur_pos_img[0]:.1f}, {cur_pos_img[1]:.1f})"
    main_ax.text(0.02, 0.98, info_text, transform=main_ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    main_ax.set_title('Floorplan with Robot State and Prediction')
    main_ax.axis('off')
    main_ax.legend()
    
    # 显示观测图像序列
    if show_obs and obs_image is not None and isinstance(axes, np.ndarray):
        if isinstance(obs_image, torch.Tensor):
            obs_np = obs_image.cpu().numpy()
            
            if obs_np.ndim == 4:  # [context_size, C, H, W]
                context_size = min(obs_np.shape[0], 5)
                positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]
                
                for i in range(context_size):
                    if i < len(positions):
                        row, col = positions[i]
                        ax = axes[row, col]
                        obs_img = obs_np[i].transpose(1, 2, 0)
                        if obs_img.max() > 1.0:
                            obs_img = obs_img / 255.0
                        ax.imshow(obs_img)
                        ax.set_title(f'Observation t-{context_size-1-i}', fontsize=10)
                        ax.axis('off')
                
                # 隐藏未使用的子图
                for i in range(context_size, len(positions)):
                    row, col = positions[i]
                    axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def transform_trajectory_to_image_coords(trajectory_global, ori_size, image_resize_size):
    """
    将全局坐标的轨迹点转换到图像坐标
    
    Args:
        trajectory_global (np.ndarray): 全局坐标的轨迹点 [N, 2]
        ori_size (float): 原始图像尺寸
        image_resize_size (Tuple[int, int]): 调整后的图像尺寸 [width, height]
    
    Returns:
        np.ndarray: 图像坐标的轨迹点 [N, 2]
    """
    w0 = ori_size
    h0 = ori_size
    
    # 第一步：全局坐标转换到原始图像像素坐标
    trajectory_pixel = trajectory_global * 100 + np.array([w0 / 2, h0 / 2])
    
    # 第二步：缩放到调整后的图像尺寸
    trajectory_resized = trajectory_pixel * image_resize_size[0] / w0
    
    return trajectory_resized

test_config_path = 'test.yaml'
with open(test_config_path, "r") as f:
    config = yaml.safe_load(f)
scene_name = 'Eudora_0'
test_dir = 'datasets/scenes_117/test'
scene_dir = os.path.join(test_dir, scene_name)
trav_folder = 'datasets/trav_maps'
data_folder = 'datasets/scenes_117/test'
model_path = 'checkpoints/ema_9.pth'
floor_shapes_ori =  np.load(os.path.join(trav_folder, "floor_shapes.npy"), allow_pickle=True).item()
image_size = config["image_size"]



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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
model.to(device)


metric_waypoint_spacing = config["metric_waypoint_spacing"]
waypoint_spacing = config["waypoint_spacing"]
# device = 'cpu'

transform = ([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform = transforms.Compose(transform)

viz_dir = "visualization_test"
os.makedirs(viz_dir, exist_ok=True)
for name in os.listdir(scene_dir):
    full_path = os.path.join(scene_dir, name)
    floorplan_path = os.path.join(scene_dir, 'floorplan.png')
    
    if os.path.isdir(full_path) and name.startswith("traj_"):
        number = name.split("_")[1]     
        traj_data = np.load(os.path.join(full_path, 'traj_' + number + '.npy'))  # (N, 5), x, y, yaw, collision, stop
        goal_pos_meter = traj_data[-2,:2].copy()
        
        # 遍历有shortest path的
        max_id = get_last_shortest_path_index(full_path)
        for i in range(5, max_id + 1, 5):
            img_path = os.path.join(full_path, '000' + str(i) + '.png')
            cur_pos_meter = traj_data[i, :2].copy()
            cur_heading_meter = traj_data[i, 2:].copy()

            # get shortest path
            shortest_path_traj_path = os.path.join(full_path, 'shortest_paths', f'shortest_path_for_{i:05d}.npy')
            cur_shortest_path = np.load(shortest_path_traj_path)  # (M, 3), x, y, yaw
            cur_shortest_path_xy = cur_shortest_path[:, :2]
            
            cur_heading_metric, obs_image, floorplan_image, cur_pos_resized, goal_pos_resized, cur_ori_resized = get_input_data(cur_pos_meter, goal_pos_meter, cur_heading_meter.copy(), img_path, floorplan_path, 'traj_' + number, scene_name, i)
            # print(cur_pos_meter, goal_pos_meter)
            cur_pos_b = cur_pos_meter[np.newaxis, :]
            cur_heading_b = cur_heading_meter[np.newaxis, :]
            goal_pos_b = goal_pos_meter[np.newaxis, :]
            floorplan_image = floorplan_image.unsqueeze(0)
            floorplan_ary = np.array(floorplan_image[0].permute(1, 2, 0).cpu())
            floorplan_ary = np.concatenate([floorplan_ary, 255*np.ones((*floorplan_ary.shape[:2],1), dtype=np.uint8)], axis=-1)
            action = execute_model(
              model = model,
              cur_pos = cur_pos_b,
              cur_heading = cur_heading_b,
              goal_pos = goal_pos_b,
              cur_obs = obs_image,
              floorplan = floorplan_image,
              metric_waypoint_spacing = metric_waypoint_spacing,
              waypoint_spacing = waypoint_spacing,
              transform = transform,
              device = device,
              noise_scheduler = noise_scheduler,
              floorplan_ary = floorplan_ary,
              log_add = 'execute_log',
          )
            # 添加可视化调用
            visualize_robot_inference_with_coords(
                cur_pos=cur_pos_meter,
                goal_pos=goal_pos_meter,
                cur_ori=cur_heading_meter,
                cur_pos_resized=cur_pos_resized,
                goal_pos_resized=goal_pos_resized,
                cur_ori_resized=cur_ori_resized,
                floorplan_image=floorplan_image,
                obs_image=obs_image,
                predicted_action=action,
                trajectory_name=f'traj_{number}',
                time_step=i,
                to_global_coords_func=to_global_coords,  # 使用你的函数
                save_path=os.path.join(viz_dir, f"traj_{number}, inference_step_{i}.png"),
                show_obs=True,
                cur_shortest_path=cur_shortest_path_xy,
            )
