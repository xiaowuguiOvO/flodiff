import pybullet as p
import numpy as np

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
    # 判断距离是否小于阈值
    if distance < threshold:
        return True
    else:
        return False

    
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
        if abs(ang_error) > np.pi / 2:  # 大于90度
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

