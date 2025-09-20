import os
import time
from igibson.envs.igibson_env import iGibsonEnv
import logging
from flona_agent import FloNaAgent
import training.train_utils as train_utils
import yaml
import numpy as np
import cv2
import pybullet as p
from training.train_utils import *
from test_utils import *
from results.data_collector import DataCollector
np.set_printoptions(precision=2, suppress=True)

PREDICT_INTERVAL = 5 # 预测超时时间
GOAL_POINT_NUM = 16
MATRIX_WAYPOINT_SPACING = 0.045
WAYPOINT_SPACING = 1.0
image_size = (96, 96)
DECISION_GOAL_POINT_INDEX = 8
floor_shapes_ori =  np.load(os.path.join("datasets/trav_maps", "floor_shapes.npy"), allow_pickle=True).item()
viz_dir = "visualization_sim"
data_save_dir = "results/res"

def main(headless=False, num_episodes=10, num_steps=200, scene_config_path=None, model_config_path=None):
    
    os.makedirs(data_save_dir, exist_ok=True)
    
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
    
    agent = FloNaAgent(model_path=model_path, model_config=model_config, scene_config=scene_config, metric_waypoint_spacing=MATRIX_WAYPOINT_SPACING, waypoint_spacing=WAYPOINT_SPACING)
    scene_name = scene_config["scene_id"] + "_" + str(scene_config["floor_num"])
    
    navigable_map_path = os.path.join(f"iGibson/igibson/data/g_dataset/{scene_config['scene_id']}", f"floor_trav_{scene_config['floor_num']}_v3.png")
    navigable_map = Image.open(navigable_map_path).resize((96, 96))


    
    for episode in range(num_episodes):
        print(f"--- Episode: {episode + 1} ---")
        
        collector = DataCollector(
            save_dir=data_save_dir,
            scene_id=scene_config["scene_id"],
            floor_num=scene_config["floor_num"],
            episode_num=episode
        )
        
        # 重置环境，获取初始观察值
        observation = env.reset()
        floorplan_img_path = os.path.join(scene_config["scene_path"], scene_config["scene_id"], 'floorplan.png')
        print(f"floorplan_img_path: {floorplan_img_path}")
        floorplan_img = Image.open(floorplan_img_path)

        IS_DECISION_FLAG = True
        
        # init state
        obs_img = observation["rgb"]
        robot_pos = env.robots[0].get_position()[:2]  # ground truth
        target_pos = env.task.target_pos[:2].copy()  # ground truth
        robot_ori_point = robot_pos # init ori point
        next_goal_point = None
        trajectory = np.zeros((32, 2))
        next_goal_point_idx = 0
        actions_meter_global = None
        FLOOR_Z = env.task.target_pos[2]    
        
        agent.update_vision_input(
            obs_img=obs_img,
            floorplan_img=floorplan_img,
            obs_pos=robot_pos,
            goal_pos=target_pos,
            obs_ori=robot_ori_point
            )
        
        # collision
        collision_count = 0

        
        for step in range(num_steps):
            
            first_collision_index = -1
            # collision_check
            if actions_meter_global is not None:
                nav_map_np_gray = np.array(navigable_map.convert('L'))
                trajectory_img_coords = transform_trajectory_to_image_coords(actions_meter_global, floor_shapes_ori[scene_name], image_size)
                if trajectory_img_coords is not None:
                    map_height, map_width = nav_map_np_gray.shape[:2]
                    for i, point in enumerate(trajectory_img_coords):
                        px, py = int(round(point[0])), int(round(point[1]))
                        # 检查是否越界
                        if not (0 <= px < map_width and 0 <= py < map_height):
                            first_collision_index = i
                            break  # 越界即碰撞
                        # 检查是否撞到障碍物 (非白色)
                        pixel_value = nav_map_np_gray[py, px]
                        if pixel_value < 255:
                            first_collision_index = i
                            break # 撞到障碍物
            
            will_collision_flag = 1 if first_collision_index != -1 and first_collision_index <= DECISION_GOAL_POINT_INDEX else 0
            
            # if collision
            if will_collision_flag:
                # print(f'Collision detected at episode {episode}, step {step}, trajectory index {first_collision_index}')            
                # replanning and rotation 45 degree
                IS_DECISION_FLAG = True
                collision_count += 1
                env.robots[0].set_rpy([0, 0, robot_yaw - math.pi/4])
                # continue  
            
            # take action
            action = [0, 0]
            # directly set robot yaw and pos
            if next_goal_point is not None and will_collision_flag != 1:
                env.robots[0].set_position([next_goal_point[0], next_goal_point[1], FLOOR_Z])
                current_pos = next_goal_point
                look_at_idx = next_goal_point_idx + 8
                if look_at_idx < len(trajectory):
                    look_at_point = trajectory[look_at_idx]
                    direction_vector = look_at_point - current_pos

                    if np.linalg.norm(direction_vector) > 1e-6:
                        new_yaw = np.arctan2(direction_vector[1], direction_vector[0])
                        env.robots[0].set_rpy([0, 0, new_yaw])
            ##### end control robot to next goal point   #####
            
            state, reward, done, info = env.step(action)
            
            # get observation
            robot_pos = env.robots[0].get_position()[:2] # ground truth
            robot_yaw = env.robots[0].get_rpy()[2]
            direction_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            robot_ori_point = robot_pos + direction_vector
            target_pos = env.task.target_pos[:2].copy()

            obs_img = state["rgb"]
                        
            floorplan_img_resize, cur_pos_resize, goal_pos_resize, cur_ori_resize = img_to_data_and_point_transfer(
                img=floorplan_img,
                ori_size=floor_shapes_ori[scene_name],
                image_resize_size=(96, 96),
                cur_pos=robot_pos.copy(),
                goal_pos=target_pos.copy(),
                cur_ori=robot_ori_point.copy()
            )

            agent.update_vision_input(
                obs_img=obs_img,
                floorplan_img=floorplan_img_resize,
                obs_pos=robot_pos.copy(),
                goal_pos=target_pos.copy(),
                obs_ori=robot_ori_point.copy()
                )
            
            # check if arrive
            if next_goal_point_idx == DECISION_GOAL_POINT_INDEX:
                IS_ARRIVE_FLAG = False
                IS_DECISION_FLAG = True
                    
            if IS_DECISION_FLAG:
                last_predict_time = time.time()
                # predict
                output = agent.get_action(agent.obs_img_queue, agent.floorplan_img, agent.obs_pos, agent.goal_pos, agent.obs_ori, MATRIX_WAYPOINT_SPACING, WAYPOINT_SPACING)
                actions = output["actions"].mean(dim=0)
                # print(actions.shape)
                actions_normed_global = to_global_coords(actions.cpu().numpy(), agent.obs_pos, agent.obs_ori)
                actions_meter_global = actions_normed_global
                actions_meter_global = actions_normed_global * MATRIX_WAYPOINT_SPACING * WAYPOINT_SPACING

                trajectory = actions_meter_global[:]    
                # draw_predicted_trajectory(trajectory, base_z=FLOOR_Z+0.02)
                
                IS_DECISION_FLAG = False
                next_goal_point_idx = 0
        
                # load navigable map
                

                visualize_robot_inference_with_coords(
                    cur_pos=agent.obs_pos,
                    goal_pos=agent.goal_pos,
                    cur_ori=agent.obs_ori,
                    cur_pos_resized=cur_pos_resize,
                    goal_pos_resized=goal_pos_resize,
                    cur_ori_resized=cur_ori_resize,
                    floorplan_image=floorplan_img_resize,
                    obs_image=agent.obs_img_queue,
                    predicted_action=actions_meter_global,
                    trajectory_name=f'traj_{step}',
                    time_step=step,
                    save_path=os.path.join(viz_dir, f"episode_{episode}, inference_step_{step}.png"),
                    show_obs=True,
                    # cur_shortest_path=cur_shortest_path_xy,
                    floor_shapes_ori=floor_shapes_ori,
                    scene_id=scene_config["scene_id"],
                    scene_floor=scene_config["floor_num"],
                    image_size=image_size,
                    navigable_map_image=navigable_map
                )
                
            collector.collect_step_data(
                robot_pos=np.array([robot_pos[0], robot_pos[1]]),
                ref_point=np.array([robot_ori_point[0], robot_ori_point[1]]),
                is_collision=will_collision_flag
                )
            #####   control robot to next goal point   #####
            # directly set robot to goal point
            decision_goal_point = trajectory[DECISION_GOAL_POINT_INDEX]
            next_goal_point = trajectory[next_goal_point_idx]
            next_goal_point_idx += 1



            if done:
                print(f"Episode ended at step {step + 1} with reward {reward}.")
                break
        
        collector.save_trajectory()
        
        if not done and num_episodes > 1: 
                print(f"Episode {episode + 1} arrive max step ({num_steps})。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    run_headless = False       

    model_path = "checkpoints\ema_9.pth" # 模型路径
    scene_config_path = "test/load_igibson_scene.yaml"
    model_config_path = 'flona.yaml'
    model_config = None
    
    main(headless=run_headless, num_episodes=15, num_steps=200, scene_config_path=scene_config_path, model_config_path=model_config_path)
