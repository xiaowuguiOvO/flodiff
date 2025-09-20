import math # 确保导入 math 模块
import os
import numpy as np
class DataCollector:
    """一个用于在仿真中采集和保存轨迹数据的类。"""
    def __init__(self, save_dir, scene_id, floor_num, episode_num):
        """
        初始化采集器。

        Args:
            save_dir (str): 数据保存的根目录。
            scene_id (str): 当前场景ID，用于构建文件名。
            floor_num (int): 当前楼层号，用于构建文件名。
            episode_num (int): 当前的回合编号，用于构建文件名。
        """
        self.save_dir = save_dir
        self.trajectory_data = []
        
        # 构建符合评估脚本要求的文件名
        # 注意：这里的 'traj_name' 是示例，您可能需要根据实际任务替换
        scene_floor = f"{scene_id}_{floor_num}"
        traj_name = f"episode_{episode_num}" # 使用 episode 编号作为轨迹名
        
        # 确保场景和轨迹的子目录存在
        self.output_dir = os.path.join(self.save_dir)
        # os.makedirs(self.output_dir, exist_ok=True)
        
        self.output_path = os.path.join(self.output_dir, f"{traj_name}.txt")

    def collect_step_data(self, robot_pos, ref_point, is_collision):
        """
        采集单个时间步的数据。

        Args:
            robot_pos (np.array): 机器人当前的 [x, y] 位置。
            ref_point (np.array): 代表朝向的参考点 [ref_x, ref_y]。
            is_collision (int): 当前步骤是否发生碰撞 (1 for yes, 0 for no)。
        """
        # 按照 [RobotX, RobotY, Ref_X, Ref_Y, Collision_Flag] 格式组织数据
        current_timestep_data = [
            robot_pos[0],
            robot_pos[1],
            ref_point[0],
            ref_point[1],
            is_collision
        ]
        self.trajectory_data.append(current_timestep_data)

    def save_trajectory(self):
        """
        将采集到的整条轨迹数据保存到.txt文件。
        """
        if not self.trajectory_data:
            print("警告：没有采集到任何轨迹数据，不会创建文件。")
            return

        # 将列表转换为Numpy数组并保存
        np.savetxt(self.output_path, np.array(self.trajectory_data), fmt='%.2f')
        # print(f"轨迹数据已成功保存至: {self.output_path}")