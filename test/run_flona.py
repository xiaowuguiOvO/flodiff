import os
import time
from igibson.envs.igibson_env import iGibsonEnv
import logging
import FloNaAgent
import training.train_utils as train_utils
import yaml
import numpy as np
import cv2
import pybullet as p
from training.train_utils import *
from test_utils import *
np.set_printoptions(precision=2, suppress=True)

PREDICT_INTERVAL = 5 # 预测超时时间
GOAL_POINT_NUM = 16

def main(headless=False, num_episodes=10, num_steps=200, scene_config_path=None, model_config_path=None):
    
    env_mode = "headless" if headless else "gui_interactive"
    env = iGibsonEnv(
        config_file=scene_config_path,
        mode=env_mode,
        use_pb_gui=True
    )

    # load scene config and model config
    with open(scene_config_path, 'r') as f:
        scene_config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    agent = FloNaAgent.FloNaAgent(model_path=model_path, config=model_config)

    for episode in range(num_episodes):
        print(f"--- Episode: {episode + 1} ---")
        # 重置环境，获取初始观察值
        observation = env.reset()
        floorplan_img_path = os.path.join(scene_config["scene_path"], scene_config["scene_id"], 'floor_0.png')
        print(f"floorplan_img_path: {floorplan_img_path}")
        floorplan_img = cv2.imread(floorplan_img_path)
        action = [0, 0]
        prev_line_ids = []
        # pd controller
        pd = PDController(Kp_lin=1, Kd_lin=0.0, Kp_ang=0.5, Kd_ang=0.1)
        IS_ARRIVE_FLAG = True
        IS_DECISION_FLAG = True
        goal_point_idx = 0
        goal_point = np.array([0, 0]) # 目标点
        prev_time = time.time()
        last_predict_time = prev_time - PREDICT_INTERVAL  # 强制第一次立即预测
        
        # init state
        obs_img = observation["rgb"]
        robot_pos = env.robots[0].get_position()[:2]  # ground truth
        robot_ori = env.robots[0].get_rpy()[2]
        obs_ori = np.array([np.sin(robot_ori), np.cos(robot_ori)])
        target_pos = env.task.target_pos[:2].copy()  # ground truth
        agent.update_vision_input(
            obs_img=obs_img,
            floorplan_img=floorplan_img,
            obs_pos=robot_pos,
            goal_pos=target_pos,
            obs_ori=obs_ori
            )
        
        # collision
        collision_count = 0
        monitor = CollisionMonitor(env.robots[0], normal_threshold=0.3, cooldown_steps=20)
        for step in range(num_steps):

            # take action
            state, reward, done, info = env.step(action)
            

            # get observation
            robot_pos = env.robots[0].get_position()[:2] # ground truth
            robot_ori = env.robots[0].get_rpy()[2] # ground truth
            obs_ori = np.array([np.sin(robot_ori), np.cos(robot_ori)])
            target_pos = env.task.target_pos[:2].copy()
            FLOOR_Z = env.task.target_pos[2]
            obs_img = state["rgb"]
            
            agent.update_vision_input(
                obs_img=obs_img,
                floorplan_img=floorplan_img,
                obs_pos=robot_pos,
                goal_pos=target_pos,
                obs_ori=obs_ori
                )
            if monitor.update(step):
                print(f"Step {step}: Collision detected.")
                # 当前姿态
                pos = env.robots[0].get_position()      # [x, y, z]
                rpy = list(env.robots[0].get_rpy())    # [roll, pitch, yaw]
                # 顺时针 45°（注意 PyBullet 的正 yaw 是逆时针，这里减号表示顺时针）
                rpy[2] = rpy[2] - math.pi/4
                # 直接“瞬移”到新的朝向（保持位置不变）
                env.robots[0].set_rpy(rpy)
                # 标记强制重新决策
                IS_DECISION_FLAG = True
                # 清空旧的 trajectory，保证下次规划使用新航向
                trajectory = np.zeros((GOAL_POINT_NUM, 2))
                # 跳过下面的 PD 控制，直接进入下一步循环
                continue
            
            # check if arrive
            if check_is_arrive(robot_pos, goal_point, threshold=0.4):
                # print("update state")

                IS_ARRIVE_FLAG = False
                # print(f"Robot arrived at the target position: {target_pos}.")
                IS_DECISION_FLAG = True

            # check time interval for decision making
            if not IS_ARRIVE_FLAG and (time.time() - last_predict_time >= PREDICT_INTERVAL):
                IS_DECISION_FLAG = True
                goal_point_idx = 0
                IS_ARRIVE_FLAG = False
                print("over time decision")

            if IS_DECISION_FLAG:

                last_predict_time = time.time()
                # predict
                metric_waipoint_spacing = 0.1
                waypoint_spacing = 1.0
                output = agent.get_action(agent.obs_img_queue, agent.floorplan_img, agent.obs_pos, agent.goal_pos, agent.obs_ori, metric_waipoint_spacing, waypoint_spacing)
                actions = output["actions"].mean(dim=0)
                # print(actions.shape)
                actions_normed_global = to_global_coords(actions.cpu().numpy(), agent.obs_pos, agent.obs_ori)
                actions_meter_global = actions_normed_global
                actions_meter_global = actions_normed_global * metric_waipoint_spacing * waypoint_spacing
                # print(actions_normed_global.shape)
                # print(actions)
                
                trajectory = actions_meter_global[:16]
                draw_predicted_trajectory(trajectory, base_z=FLOOR_Z+0.02)
                
                IS_DECISION_FLAG = False
                goal_point_idx = 0
            
            # print(robot_pos, goal_point)
            # pd control
            now = time.time()
            dt = now - prev_time
            prev_time = now
            goal_point = trajectory[GOAL_POINT_NUM - 1]
            look_ahead_point = compute_look_ahead_point(trajectory, robot_pos, ahead_dis=0.3)
            # print(look_ahead_point)
            # print(action[0])
            action = pd.compute(robot_pos, robot_ori, look_ahead_point, dt)
            # action = [0, 0]
            # action[0] = 0
            # print(robot_ori)
            # print(f'goal_linear_speed: {action[0]}')
            # print(f'cur_angular_speed, goal_angular_speed: {robot_ori}, {action[1]}')
            # action = [0,0]
            # print(trajectory)
            if done:
                print(f"Episode ended at step {step + 1} with reward {reward}.")
                break
        
        if not done and num_episodes > 1: 
                print(f"Episode {episode + 1} arrive max step ({num_steps})。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    run_headless = False       

    model_path = "checkpoints\ema_67_2.pth" # 模型路径
    scene_config_path = "test/load_igibson_scene.yaml"
    model_config_path = 'flona.yaml'
    model_config = None
    
    main(headless=run_headless, num_episodes=15, num_steps=200, scene_config_path=scene_config_path, model_config_path=model_config_path)
