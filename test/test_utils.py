import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import torch
from typing import Tuple
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

fig, ax_list = None, None

np.set_printoptions(precision=2, suppress=True)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
CYAN = np.array([0, 1, 1])
MAGENTA = np.array([1, 0, 1])

def draw_predicted_trajectory(trajectory, base_z=0.0):
    """
    每次在 PyBullet 中绘制一条新的预测轨迹，并在绘制前清除之前由此函数绘制的内容。
    :param trajectory: np.array of shape (N, 2), 包含 [x, y] 点.
    :param base_z: float, 轨迹应该围绕其绘制的Z轴高度.
    """
    # 使用函数属性来存储上一次绘制的 debug item IDs
    # 如果属性不存在，则初始化为空列表
    if not hasattr(draw_predicted_trajectory, "_previous_debug_items"):
        draw_predicted_trajectory._previous_debug_items = []

    # 1. 清除上一次绘制的 debug items
    for item_id in draw_predicted_trajectory._previous_debug_items:
        try:
            # 尝试移除 PyBullet 中的 debug item
            # PyBullet 的 removeUserDebugItem 可能会在 ID 无效时打印错误或无操作
            # 而不是总是抛出 Python 异常，因此 try-except 可能不是必须的
            # 但为了安全起见可以保留
            p.removeUserDebugItem(item_id)
        except Exception: # 捕获任何可能的错误，例如 item 已被其他方式移除
            pass
    draw_predicted_trajectory._previous_debug_items.clear() # 清空列表，准备存储新的 IDs

    # ---- 开始绘制新的轨迹 ----

    # 确保 trajectory 是一个可以迭代的二维点集合
    if trajectory is None or len(trajectory) == 0:
        return # 如果没有轨迹数据，则不执行任何操作

    # 使用传入的 base_z，并可能加上一个小偏移量，使线条在地面之上可见
    line_z = float(base_z) + 0.02  # 轨迹线本身的Z高度
    marker_base_z = float(base_z) + 0.02 # 标记点也围绕这个Z高度

    # 将2D轨迹点转换为3D点，确保所有坐标都是浮点数
    try:
        pts3d = [[float(x), float(y), line_z] for x, y in trajectory]
    except (TypeError, ValueError) as e:
        print(f"错误：轨迹数据格式不正确: {e}. 期望 Nx2 的点集。")
        return


    current_items_drawn = [] # 存储本次调用绘制的 item IDs

    # 绘制轨迹线 (绿色)
    if len(pts3d) > 1: # 至少需要两个点才能画线
        for i in range(len(pts3d) - 1):
            try:
                line_id = p.addUserDebugLine(
                    pts3d[i], pts3d[i + 1],
                    lineColorRGB=[0, 1, 0],  # 绿色
                    lineWidth=2
                    # lifeTime 参数已移除
                )
                current_items_drawn.append(line_id)
            except Exception as e:
                print(f"绘制轨迹线时出错: {e}")


    # 轨迹点标记（红色小竖线）
    for pt_3d_coords in pts3d: # pt_3d_coords 是 [x, y, line_z]
        try:
            marker_id = p.addUserDebugLine(
                [pt_3d_coords[0], pt_3d_coords[1], marker_base_z - 0.02],  # 标记的底部
                [pt_3d_coords[0], pt_3d_coords[1], marker_base_z + 0.02],  # 标记的顶部
                lineColorRGB=[1, 0, 0],  # 红色
                lineWidth=3
                # lifeTime 参数已移除
            )
            current_items_drawn.append(marker_id)
        except Exception as e:
            print(f"绘制轨迹点标记时出错: {e}")


    # 更新函数属性，以便下次调用时清除这些 items
    draw_predicted_trajectory._previous_debug_items = current_items_drawn

def check_is_arrive(robot_pos, target_pos, threshold=0.1):
    """
    检查机器人是否到达目标位置。

    Args:
        robot_pos (array-like): 机器人的当前位置 (例如 [x, y] 或 [x, y, z])。
                                可以是列表、元组或 NumPy 数组。
        target_pos (array-like): 目标位置 (例如 [x, y] 或 [x, y, z])。
                                 可以是列表、元组或 NumPy 数组。
        threshold (float, optional): 判断是否到达的距离阈值。默认为 0.1。

    Returns:
        bool: 如果机器人与目标点的距离小于阈值，则返回 True，否则返回 False。
    """
    # 将输入转换为 NumPy 数组，以便于进行向量运算
    robot_pos_np = np.array(robot_pos)
    target_pos_np = np.array(target_pos)

    # 确保两个位置的维度相同
    if robot_pos_np.shape != target_pos_np.shape:
        raise ValueError("机器人位置和目标位置的维度必须相同。")

    # 计算欧几里得距离
    # np.linalg.norm(a - b) 计算了向量 a 和 b 之间的欧几里得距离
    distance = np.linalg.norm(robot_pos_np - target_pos_np)
    # print(distance)
    # 判断距离是否小于阈值
    if distance < threshold:
        return True
    else:
        return False


def compute_look_ahead_point(trajectory, robot_pos, ahead_dis=0.5):
    # 修改检查 trajectory 是否有效的方式
    if trajectory is None:
        print("错误：轨迹为 None。")
        return None
    
    # 对于 NumPy 数组，检查其大小 (size) 或 第一个维度 (shape[0])
    # isinstance(trajectory, np.ndarray) 用于确认它确实是 NumPy 数组
    if isinstance(trajectory, np.ndarray):
        if trajectory.size == 0: # 数组中没有任何元素
            print("错误：轨迹 NumPy 数组为空 (size is 0)。")
            return None
        # 或者检查行数，如果 trajectory.ndim >= 1 (至少是一维数组)
        # if trajectory.ndim == 0: # 标量 NumPy 对象，不是轨迹
        #     print("错误：轨迹是标量 NumPy 对象。")
        #     return None
        if trajectory.shape[0] == 0: # 没有路径点
             print("错误：轨迹 NumPy 数组中没有路径点 (shape[0] is 0)。")
             return None
    elif not trajectory: # 对于其他序列类型 (如 Python 列表)
        print("错误：轨迹列表为空。")
        return None

    # 确保轨迹至少有一个点可以作为回退
    # （如果只有一个点，下面的循环不会执行，但会直接返回该点或在末尾返回）
    if (isinstance(trajectory, np.ndarray) and trajectory.shape[0] == 1) or \
       (not isinstance(trajectory, np.ndarray) and len(trajectory) == 1):
        return trajectory[0] if isinstance(trajectory, list) else tuple(trajectory[0])


    look_ahead_point = None
    
    # trajectory 是 NumPy 数组时，len(trajectory) 也能正确给出点的数量 (行数)
    num_points = trajectory.shape[0] if isinstance(trajectory, np.ndarray) else len(trajectory)

    for i in range(num_points - 1):
        p1 = trajectory[i]      # 对于 NumPy 数组，p1 是一行，例如 np.array([-2.3, 0.62])
        p2 = trajectory[i+1]

        # 后续的 p1[0], p1[1] 等索引对于 NumPy 数组的行是有效的
        seg_dx = p2[0] - p1[0]
        seg_dy = p2[1] - p1[1]

        robot_to_p1_dx = p1[0] - robot_pos[0]
        robot_to_p1_dy = p1[1] - robot_pos[1]
        
        a = seg_dx * seg_dx + seg_dy * seg_dy
        b = 2 * (robot_to_p1_dx * seg_dx + robot_to_p1_dy * seg_dy)
        c = robot_to_p1_dx * robot_to_p1_dx + robot_to_p1_dy * robot_to_p1_dy - ahead_dis * ahead_dis

        if abs(a) < 1e-9:
            continue

        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)

            current_segment_valid_ts = []
            if 0 <= t1 <= 1:
                current_segment_valid_ts.append(t1)
            # 确保 t2 不同于 t1 才添加，或者 t1 无效时添加 t2
            if 0 <= t2 <= 1 and (abs(t1 - t2) > 1e-9 or not (0 <= t1 <= 1)):
                 current_segment_valid_ts.append(t2)
            
            if current_segment_valid_ts:
                chosen_t = max(current_segment_valid_ts)
                # 结果点应为元组
                look_ahead_point = (p1[0] + chosen_t * seg_dx, 
                                    p1[1] + chosen_t * seg_dy)
    
    if look_ahead_point is None and num_points > 0:
        # 如果没有找到交点，通常选择轨迹的最后一个点
        # 确保返回的是元组
        last_point_data = trajectory[-1]
        look_ahead_point = tuple(last_point_data) 
        
    return look_ahead_point

    
    
class PDController:
    def __init__(self, Kp_lin=1.0, Kd_lin=0.1, Kp_ang=3.0, Kd_ang=0.2):
        self.Kp_lin = Kp_lin
        self.Kd_lin = Kd_lin
        self.Kp_ang = Kp_ang
        self.Kd_ang = Kd_ang
        self.last_lin_error = 0.0
        self.last_ang_error = 0.0

    def reset(self):
        self.last_lin_error = 0.0
        self.last_ang_error = 0.0

    def compute(self, current_pos, current_yaw, target_pos, dt):
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        diff = target_pos - current_pos  # [dx, dy]

        # 目标方向角度
        target_theta = np.arctan2(diff[1], diff[0])
        
        # 计算并归一化角度误差到 (-pi, pi]
        # 您原来的方法是有效的，也可以用下面的标准方法：
        raw_ang_diff = target_theta - current_yaw
        ang_error = np.arctan2(np.sin(raw_ang_diff), np.cos(raw_ang_diff))
        # 或者保持您原来的:
        # ang_error = (target_theta - current_yaw + np.pi) % (2 * np.pi) - np.pi

        lin_error = np.linalg.norm(diff)

        # 防止 dt 过小或为零导致的不稳定或除零错误
        safe_dt = dt if dt > 1e-6 else 1e-6 

        # 计算误差导数
        lin_deriv = (lin_error - self.last_lin_error) / safe_dt
        ang_deriv = (ang_error - self.last_ang_error) / safe_dt

        # 首先计算无约束的角速度
        omega_unconstrained = self.Kp_ang * ang_error + self.Kd_ang * ang_deriv
        
        # 计算无约束的线速度
        v_unconstrained = self.Kp_lin * lin_error + self.Kd_lin * lin_deriv
        
        # 应用约束：如果角度差过大，则线速度为0 (原地转向)
        v_final = v_unconstrained
        if abs(ang_error) > np.pi / 4:  # 大于90度
            v_final = 0.0
            
        # 更新上一次的误差记录
        # 重要的是，last_lin_error 应该存储实际的距离误差，
        # 而不是被条件修改后的误差，以便D项能正确反映距离的变化趋势。
        self.last_lin_error = lin_error 
        self.last_ang_error = ang_error

        # print(ang_error, omega_unconstrained)
        # print(np.array([current_yaw, target_theta, omega_unconstrained, ang_error]))
        # print(f"current_yaw: {current_yaw:.2f},target_yaw: {target_theta:.2f} ang_error: {ang_error:.2f}, omega: {omega_unconstrained:.2f}")
        return [v_final, -omega_unconstrained]

class CollisionMonitor:
    def __init__(self, robot, normal_threshold=0.3, cooldown_steps=10):
        """
        :param robot: iGibson 中的机器人实例（比如 env.robots[0]）
        :param normal_threshold: 法线 z 分量阈值 (|nz| > threshold 则视为地面支撑)
        :param cooldown_steps: 冷却步数，检测到一次碰撞后，接下来 cooldown_steps 步内不再重复计数
        """
        self.robot = robot
        self.normal_threshold = normal_threshold
        self.cooldown_steps = cooldown_steps

        self.last_collision_step = -cooldown_steps
        self.collision_count = 0

    def update(self, current_step):
        """
        每个仿真步调用一次，返回这一步是否为新碰撞：
        :param current_step: 当前仿真步编号（整型）
        :return: bool，True 表示这一步计入了一次新碰撞
        """
        # 检测是否有横向碰撞
        collided = False
        for body_id in self.robot.get_body_ids():
            cps = p.getContactPoints(bodyA=body_id)
            for cp in cps:
                nx, ny, nz = cp[7]  # contactNormalOnB
                if abs(nz) <= self.normal_threshold:
                    collided = True
                    break
            if collided:
                break

        # 如果撞了，且已过冷却期，就计数
        if collided and (current_step - self.last_collision_step) >= self.cooldown_steps:
            self.collision_count += 1
            self.last_collision_step = current_step
            return True

        return False
    
# def visualize_diffusion_action_distribution(
#     ema_model: nn.Module,
#     noise_scheduler: DDPMScheduler,
#     batch_obs_images: torch.Tensor,
#     batch_floorplan_images: torch.Tensor,
#     batch_viz_obs_images: torch.Tensor,
#     batch_viz_floorplan_images: torch.Tensor,
#     batch_action_label: torch.Tensor,
#     batch_distance_labels: torch.Tensor,
#     batch_goal_pos: torch.Tensor,
#     batch_curr_pos: torch.Tensor,
#     batch_curr_ori: torch.Tensor,
#     batch_goal_pos_local: torch.Tensor,
#     batch_goal_pos_resized: torch.Tensor,
#     batch_curr_pos_resized: torch.Tensor,
#     batch_curr_ori_resized: torch.Tensor,
#     device: torch.device,
#     type: str,
#     project_folder: str,
#     epoch: int,
#     num_images_log: int,
#     num_samples: int = 30,
#     use_swanlab: bool = True,
# ):
#     """Plot samples from the exploration model."""

#     visualize_path = os.path.join(
#         project_folder,
#         "visualize",
#         type,
#         f"epoch{epoch}",
#         "action_sampling_prediction",
#     )
#     if not os.path.isdir(visualize_path):
#         os.makedirs(visualize_path)

#     max_batch_size = batch_obs_images.shape[0]
#     num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_floorplan_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
#     batch_obs_images = batch_obs_images[:num_images_log]
#     batch_floorplan_images = batch_floorplan_images[:num_images_log]
#     batch_action_label = batch_action_label[:num_images_log]
#     batch_goal_pos = batch_goal_pos[:num_images_log]
#     batch_curr_pos = batch_curr_pos[:num_images_log]
#     batch_curr_ori = batch_curr_ori[:num_images_log]
#     batch_goal_pos_local = batch_goal_pos_local[:num_images_log]
#     batch_goal_pos_resized = batch_goal_pos_resized[:num_images_log]
#     batch_curr_pos_resized = batch_curr_pos_resized[:num_images_log]
#     batch_curr_ori_resized = batch_curr_ori_resized[:num_images_log]
#     swanlab_list = []

#     pred_horizon = batch_action_label.shape[1]
#     action_dim = batch_action_label.shape[2]

#     # split into batches
#     batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
#     batch_floorplan_images_list = torch.split(batch_floorplan_images, max_batch_size, dim=0)
#     actions_list = []
#     distances_list = []

#     for obs, floorplan in zip(batch_obs_images_list, batch_floorplan_images_list):
#         model_output_dict = model_output(
#             ema_model,
#             noise_scheduler,
#             obs,
#             floorplan,
#             pred_horizon,
#             action_dim,
#             num_samples,
#             batch_goal_pos,
#             batch_curr_pos,
#             batch_curr_ori,
#             device,
#         )
#         actions_list.append(to_numpy(model_output_dict['actions'])) # local, waypoints metric
#         distances_list.append(to_numpy(model_output_dict['distance']))

#     # concatenate
#     actions_list = np.concatenate(actions_list, axis=0)
#     distances_list = np.concatenate(distances_list, axis=0)

#     # split into actions per observation
#     actions_list = np.split(actions_list, num_images_log, axis=0)
#     distances_list = np.split(distances_list, num_images_log, axis=0)
#     distances_avg = [np.mean(dist) for dist in distances_list]
#     distances_std = [np.std(dist) for dist in distances_list]
#     assert len(actions_list) == len(actions_list) == num_images_log
#     np_distance_labels = to_numpy(batch_distance_labels)
#     for i in range(num_images_log):
#         fig, ax = plt.subplots(1, 3)
#         actions = actions_list[i]
#         action_label = to_numpy(batch_action_label[i])
#         traj_list = np.concatenate([
#             actions,
#             action_label[None],
#         ], axis=0)
#         traj_colors = ["red"] * len(actions) + ["magenta"]
#         traj_alphas = [0.1] * len(actions) + [1.0]

#         # make points numpy array of robot positions (0, 0) and goal positions
#         point_list = [np.array([0, 0]), to_numpy(batch_goal_pos_local[i])]
#         point_colors = ["green", "red"]
#         point_alphas = [1.0, 1.0]
#         plot_trajs_and_points(
#             ax[0],
#             traj_list,
#             point_list,
#             traj_colors,
#             point_colors,
#             traj_labels=None,
#             point_labels=None,
#             traj_alphas=traj_alphas,
#             point_alphas=point_alphas,
#             default_coloring=True, 
#         )
        
#         obs_image = to_numpy(batch_viz_obs_images[i])
#         floorplan_image = to_numpy(batch_viz_floorplan_images[i])
#         # move channel to last dimension
#         obs_image = np.moveaxis(obs_image, 0, -1)
#         floorplan_image = np.moveaxis(floorplan_image, 0, -1)
#         ax[1].imshow(obs_image)
#         ax[2].imshow(floorplan_image)

#         # set title
#         ax[0].set_title(f"diffusion action predictions")
#         ax[1].set_title(f"observation")
#         ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={distances_avg[i]:.2f}±{distances_std[i]:.2f}")
#         str_text = f'goal_resized:{batch_goal_pos_resized[i].cpu().numpy()} curr_pos_resized:{batch_curr_pos_resized[i].cpu().numpy()} curr_ori_resized:{batch_curr_ori_resized[i].cpu().numpy()}'
#         fig.text(0, 0, str_text)
        
#         # make the plot large
#         fig.set_size_inches(18.5, 10.5)
#         save_path = os.path.join(visualize_path, f"sample_{i}.png")
#         plt.savefig(save_path)
#         # swanlab_list.append(swanlab.Image(save_path))
#         plt.close(fig)
#     if len(swanlab_list) > 0 and use_swanlab:
#         swanlab.log({f"{type}_action_samples": swanlab_list}, commit=False)

# def plot_trajs_and_points(
#     ax: plt.Axes,
#     list_trajs: list,
#     list_points: list,
#     traj_colors: list = [CYAN, MAGENTA],
#     point_colors: list = [RED, GREEN],
#     traj_labels: Optional[list] = ["prediction", "ground truth"],
#     point_labels: Optional[list] = ["robot", "goal"],
#     traj_alphas: Optional[list] = None,
#     point_alphas: Optional[list] = None,
#     default_coloring: bool = True,
# ):
#     """
#     Plot trajectories and points that could potentially have a yaw.

#     Args:
#         ax: matplotlib axis
#         list_trajs: list of trajectories, each trajectory is a numpy array of shape (horizon, 2) 
#         list_points: list of points, each point is a numpy array of shape (2,)
#         traj_colors: list of colors for trajectories
#         point_colors: list of colors for points
#         traj_labels: list of labels for trajectories
#         point_labels: list of labels for points
#         traj_alphas: list of alphas for trajectories
#         point_alphas: list of alphas for points
#     """
#     assert (
#         len(list_trajs) <= len(traj_colors) or default_coloring
#     ), "Not enough colors for trajectories"
#     assert len(list_points) <= len(point_colors), "Not enough colors for points"
#     assert (
#         traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
#     ), "Not enough labels for trajectories"
#     assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

#     for i, traj in enumerate(list_trajs):
#         if traj_labels is None:
#             ax.plot(
#                 traj[:, 0], 
#                 traj[:, 1], 
#                 color=traj_colors[i],
#                 alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
#                 marker="o",
#             )
#         else:
#             ax.plot(
#                 traj[:, 0],
#                 traj[:, 1],
#                 color=traj_colors[i],
#                 label=traj_labels[i],
#                 alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
#                 marker="o",
#             )
#     for i, pt in enumerate(list_points):
#         if point_labels is None:
#             ax.plot(
#                 pt[0], 
#                 pt[1], 
#                 color=point_colors[i], 
#                 alpha=point_alphas[i] if point_alphas is not None else 1.0,
#                 marker="o",
#                 markersize=7.0
#             )
#         else:
#             ax.plot(
#                 pt[0],
#                 pt[1],
#                 color=point_colors[i],
#                 alpha=point_alphas[i] if point_alphas is not None else 1.0,
#                 marker="o",
#                 markersize=7.0,
#                 label=point_labels[i],
#             )
#     # put the legend below the plot
#     if traj_labels is not None or point_labels is not None:
#         ax.legend()
#         ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
#     ax.set_aspect("equal", "box")
    
def visualize_diffusion_action_distribution(
    model_output_action_list_abs,    # shape: (num_samples, pred_horizon, action_dim)
    model_output_action_list_local,  # shape: same as above, but in local frame
    goal_pos_abs,                    # shape: (2,)
    goal_pos_local,                  # shape: (2,)
    global_ori,                      # float, in radians
    ground_truth_dist,              # float
    model_output_dist,               # list or array of floats, shape: (num_samples,)
    obs_np                           # numpy array of shape (H, W, C)
):
    global fig, ax_list

    if fig is None or ax_list is None:
        plt.ion()
        fig, ax_list = plt.subplots(1, 3, figsize=(18, 6))

    ax_abs, ax_local, ax_obs = ax_list

    # 清空旧图像
    ax_abs.clear()
    ax_local.clear()

    num_samples = len(model_output_action_list_abs)

    # === 绝对轨迹图 ===
    for i in range(num_samples):
        traj_abs = model_output_action_list_abs[i]
        ax_abs.plot(traj_abs[:, 0], traj_abs[:, 1], color='red', alpha=0.1)

    avg_traj_abs = np.mean(model_output_action_list_abs, axis=0)
    ax_abs.plot(avg_traj_abs[:, 0], avg_traj_abs[:, 1], color='blue', linewidth=2, label='Avg Traj')
    ax_abs.scatter(goal_pos_abs[0], goal_pos_abs[1], color='green', s=100, label='Goal (abs)')

    # 起点位置
    start_x = model_output_action_list_abs[0][0, 0]
    start_y = model_output_action_list_abs[0][0, 1]
    ax_abs.scatter(start_x, start_y, color='black', s=80, label='Start (abs)')

    # 添加机器人朝向箭头
    arrow_len = 4  # 箭头长度
    dx = arrow_len * np.cos(global_ori)
    dy = arrow_len * np.sin(global_ori)
    ax_abs.quiver(start_x, start_y, dx, dy, angles='xy', scale_units='xy', scale=1, color='orange', width=0.01, label='Orientation')

    ax_abs.set_title('Absolute Trajectories')
    ax_abs.set_xlabel('X')
    ax_abs.set_ylabel('Y')
    ax_abs.axis('equal')
    ax_abs.grid(True)
    ax_abs.legend()

    # === 相对轨迹图 ===
    for i in range(num_samples):
        traj_local = model_output_action_list_local[i]
        ax_local.plot(traj_local[:, 0], traj_local[:, 1], color='red', alpha=0.1)

    avg_traj_local = np.mean(model_output_action_list_local, axis=0)
    ax_local.plot(avg_traj_local[:, 0], avg_traj_local[:, 1], color='blue', linewidth=2, label='Avg Traj')
    ax_local.scatter(goal_pos_local[0], goal_pos_local[1], color='green', s=100, label='Goal (local)')
    ax_local.scatter(0, 0, color='black', s=80, label='Start (0,0)')

    ax_local.set_title(f'Relative Trajectories\nGT Dist: {ground_truth_dist:.2f}, '
                       f'Pred: {np.mean(model_output_dist):.2f} ± {np.std(model_output_dist):.2f}')
    ax_local.set_xlabel('X')
    ax_local.set_ylabel('Y')
    ax_local.axis('equal')
    ax_local.grid(True)
    ax_local.legend()

    # obs image
    ax_obs.clear()
    img_to_show = obs_np.copy()
    ax_obs.imshow(img_to_show)
    ax_obs.set_title("Observation Image")
    ax_obs.axis("off")
    
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# 我将你的 transform_trajectory_to_image_coords 函数放在前面，以便整个代码块可以直接运行
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
    w0 = h0 = ori_size
    # 第一步：全局坐标转换到原始图像像素坐标
    trajectory_pixel = trajectory_global * 100 + np.array([w0 / 2, h0 / 2])
    # 第二步：缩放到调整后的图像尺寸
    trajectory_resized = trajectory_pixel * image_resize_size[0] / w0
    return trajectory_resized

def visualize_robot_inference_with_coords(cur_pos, goal_pos, cur_ori, cur_pos_resized, 
                                          goal_pos_resized, cur_ori_resized, floorplan_image, 
                                          navigable_map_image=None,
                                          obs_image=None, predicted_action=None, trajectory_name=None, time_step=None, 
                                          save_path=None, show_obs=True, floor_shapes_ori=None, 
                                          scene_id=None, scene_floor=None, image_size=None):
    """
    可视化机器人推理过程，可同时显示 floorplan 和 navigable map。
    如果发生碰撞，则将碰撞点及之后的轨迹点标为红色。
    """
    scene_name = f"{scene_id}_{scene_floor}"
    has_nav_map = navigable_map_image is not None
    
    # --- 1. 创建子图布局 (无变化) ---
    if show_obs and obs_image is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        floorplan_ax = axes[0, 0]
        nav_ax = axes[0, 1] if has_nav_map else None
    else:
        num_maps = 2 if has_nav_map else 1
        fig, axes = plt.subplots(1, num_maps, figsize=(9 * num_maps, 8))
        if num_maps == 1:
            floorplan_ax, nav_ax = axes, None
        else:
            floorplan_ax, nav_ax = axes[0], axes[1]

    fig.suptitle(f'Robot Inference Visualization - {trajectory_name} - Step {time_step}', fontsize=16)
    
    # --- 预处理坐标、轨迹和地图 ---
    cur_pos_img = cur_pos_resized
    goal_pos_img = goal_pos_resized
    
    trajectory_img_coords = None
    if predicted_action is not None:
        action_np = predicted_action.cpu().numpy() if isinstance(predicted_action, torch.Tensor) else np.array(predicted_action)
        trajectory_img_coords = transform_trajectory_to_image_coords(action_np, floor_shapes_ori[scene_name], image_size)

    # 为了碰撞检测和绘图，提前处理 navigable_map
    nav_map_np = None
    if has_nav_map:
        if isinstance(navigable_map_image, torch.Tensor):
            if navigable_map_image.dim() == 4: nav_map_np = navigable_map_image[0].permute(1, 2, 0).cpu().numpy()
            elif navigable_map_image.dim() == 3: nav_map_np = navigable_map_image.permute(1, 2, 0).cpu().numpy()
            else: nav_map_np = navigable_map_image.cpu().numpy()
        else:
            nav_map_np = np.array(navigable_map_image)
        if nav_map_np.max() > 1.0: nav_map_np = nav_map_np / 255.0
        # 如果是RGB图，转为灰度值用于碰撞检测
        if nav_map_np.ndim == 3 and nav_map_np.shape[2] == 3:
             nav_map_np_gray = (nav_map_np * 255).astype(np.uint8)
             nav_map_np_gray = np.dot(nav_map_np_gray[...,:3], [0.2989, 0.5870, 0.1140])
        else:
             nav_map_np_gray = (nav_map_np * 255).astype(np.uint8)


    # <<< MODIFIED: 在这里添加碰撞检测逻辑 >>>
    first_collision_index = -1  # 默认为-1，表示没有碰撞
    if has_nav_map and trajectory_img_coords is not None:
        map_height, map_width = nav_map_np_gray.shape[:2]
        for i, point in enumerate(trajectory_img_coords):
            px, py = int(round(point[0])), int(round(point[1]))

            # 检查是否越界
            if not (0 <= px < map_width and 0 <= py < map_height):
                first_collision_index = i
                break  # 越界即碰撞

            # 检查是否撞到障碍物 (非白色)
            pixel_value = nav_map_np_gray[py, px]
            if pixel_value == 0:
                first_collision_index = i
                break # 撞到障碍物
    # <<< MODIFIED END >>>

    # --- 2. 处理和绘制 Floorplan Map ---
    if isinstance(floorplan_image, torch.Tensor):
        if floorplan_image.dim() == 4: floorplan_np = floorplan_image[0].permute(1, 2, 0).cpu().numpy()
        elif floorplan_image.dim() == 3: floorplan_np = floorplan_image.permute(1, 2, 0).cpu().numpy()
        else: floorplan_np = floorplan_image.cpu().numpy()
    else:
        floorplan_np = np.array(floorplan_image)
    if floorplan_np.max() > 1.0: floorplan_np = floorplan_np / 255.0
    
    floorplan_ax.imshow(floorplan_np)
    floorplan_ax.add_patch(Circle(cur_pos_img, radius=2, color='green', alpha=0.8, label='Current'))
    floorplan_ax.add_patch(Circle(goal_pos_img, radius=2, color='red', alpha=0.8, label='Goal'))
    
    arrow_length = 15
    ori_angle = np.arctan2(cur_ori[1] - cur_pos[1], cur_ori[0] - cur_pos[0])
    arrow_end_x = cur_pos_img[0] + arrow_length * np.cos(ori_angle)
    arrow_end_y = cur_pos_img[1] + arrow_length * np.sin(ori_angle)
    floorplan_ax.add_patch(FancyArrowPatch(cur_pos_img, (arrow_end_x, arrow_end_y), arrowstyle='->', mutation_scale=20, color='blue', linewidth=2, label='Orientation'))

    if trajectory_img_coords is not None:
        for i, point in enumerate(trajectory_img_coords):
            lbl = 'Prediction' if i == 0 else None
            # 根据是否检测到碰撞来决定点的颜色
            color = 'red' if first_collision_index != -1 and i >= first_collision_index else 'cyan'
            floorplan_ax.add_patch(Circle(point, radius=0.2, color=color, alpha=0.8, label=lbl))
            
    info_text = f"Global: pos({cur_pos[0]:.2f}, {cur_pos[1]:.2f})\nImage: pos({cur_pos_img[0]:.1f}, {cur_pos_img[1]:.1f})"
    floorplan_ax.text(0.02, 0.98, info_text, transform=floorplan_ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    floorplan_ax.set_title('Floorplan View')
    floorplan_ax.axis('off')
    floorplan_ax.legend()

    # --- 3. 处理和绘制 Navigable Map ---
    if has_nav_map and nav_ax is not None:
        nav_ax.imshow(nav_map_np, cmap='gray')
        nav_ax.add_patch(Circle(cur_pos_img, radius=2, color='green', alpha=0.8, label='Current'))
        nav_ax.add_patch(Circle(goal_pos_img, radius=2, color='red', alpha=0.8, label='Goal'))
        nav_ax.add_patch(FancyArrowPatch(cur_pos_img, (arrow_end_x, arrow_end_y), arrowstyle='->', mutation_scale=20, color='blue', linewidth=2, label='Orientation'))

        # <<< MODIFIED: 同样修改此处的绘图循环 >>>
        if trajectory_img_coords is not None:
            for i, point in enumerate(trajectory_img_coords):
                lbl = 'Prediction' if i == 0 else None
                # 使用相同的逻辑决定颜色
                color = 'red' if first_collision_index != -1 and i >= first_collision_index else 'cyan'
                nav_ax.add_patch(Circle(point, radius=0.2, color=color, alpha=0.8, label=lbl))
        # <<< MODIFIED END >>>

        nav_ax.text(0.02, 0.98, info_text, transform=nav_ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        nav_ax.set_title('Navigable Map View')
        nav_ax.axis('off')
        nav_ax.legend()
    
    # --- 4. 显示观测图像序列 (无变化) ---
    if show_obs and obs_image is not None:
        # ... (此部分代码无须修改) ...
        obs_stack = torch.stack(obs_image, dim=0)
        obs_np_all = obs_stack.cpu().numpy()
        positions = [(0, 2), (1, 0), (1, 1), (1, 2)] 
        if not has_nav_map:
            positions.insert(0, (0, 1))
        context_size = min(obs_np_all.shape[0], len(positions))
        for i in range(context_size):
            row, col = positions[i]
            ax = axes[row, col] if (show_obs and obs_image is not None and (isinstance(axes, np.ndarray) and axes.ndim > 1)) else axes
            obs_img = obs_np_all[i].transpose(1, 2, 0)
            if obs_img.max() > 1.0: obs_img /= 255.0
            ax.imshow(obs_img)
            ax.set_title(f'Observation t-{context_size-1-i}')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- 5. 保存或显示 ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

def img_to_data_and_point_transfer(
    img: Image.Image,
    ori_size: float,
    image_resize_size: Tuple[int, int],
    cur_pos: np.ndarray,
    goal_pos: np.ndarray,
    cur_ori: np.ndarray
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform the input image and transfer the points to local coordinates.
    
    Args:
        img (Image.Image): PIL image object
        ori_size (float): original reference size
        image_resize_size (Tuple[int, int]): size to resize the image to [width, height]
        cur_pos (np.ndarray): current position in pixel coordinates of original image [x,y]
        goal_pos (np.ndarray): goal position in pixel coordinates of original image
        cur_ori (np.ndarray): current orientation vector (or point) in original image

    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
            - resized image as tensor
            - current position in the transformed image size coordinate
            - goal position in the same coordinate
            - current orientation in the same coordinate
    """
    w0 = ori_size
    h0 = ori_size
    w, h = img.size

    # transform positions
    cur_pos = cur_pos * 100 + np.array([w0 / 2, h0 / 2])
    goal_pos = goal_pos * 100 + np.array([w0 / 2, h0 / 2])
    cur_ori = cur_ori * 100 + np.array([w0 / 2, h0 / 2])      

    # resize
    img = img.resize(image_resize_size)
    cur_pos_in_resizeSize = cur_pos * image_resize_size[0] / w0
    goal_pos_in_resizeSize = goal_pos * image_resize_size[0] / w0
    cur_ori_in_resizeSize = cur_ori * image_resize_size[0] / w0

    # convert to tensor
    resize_img = TF.to_tensor(img)

    return resize_img, cur_pos_in_resizeSize, goal_pos_in_resizeSize, cur_ori_in_resizeSize

def check_collision_on_map(
    navigable_map: np.ndarray,
    trajectory_img_coords: np.ndarray,
    index: int
) -> bool:

    map_height, map_width = navigable_map.shape[:2]
    check_up_to = min(index, len(trajectory_img_coords) - 1)

    for i in range(check_up_to + 1):
        point = trajectory_img_coords[i]
        px, py = int(round(point[0])), int(round(point[1]))

        if not (0 <= px < map_width and 0 <= py < map_height):
            return True # 越界碰撞

        pixel_value = navigable_map[py, px] if navigable_map.ndim == 2 else navigable_map[py, px, 0]
        if pixel_value < 255:
            return True # 障碍物碰撞

    return False