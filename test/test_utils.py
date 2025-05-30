import pybullet as p
import numpy as np
import math
np.set_printoptions(precision=2, suppress=True)

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