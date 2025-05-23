import os
import time
from igibson.envs.igibson_env import iGibsonEnv
import logging
import FloNaAgent
import training.train_utils as train_utils
import yaml
import numpy as np
import cv2
def main(headless=False, short_exec_episodes=3, short_exec_steps=200, scene_config_path=None, model_config_path=None):
    
    env_mode = "headless" if headless else "gui_interactive"
    env = iGibsonEnv(
        config_file=scene_config_path,
        mode=env_mode
    )

    # load scene config and model config
    with open(scene_config_path, 'r') as f:
        scene_config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    agent = FloNaAgent.FloNaAgent(model_path=model_path, config=model_config)
    num_episodes = 1 if short_exec_episodes is None else short_exec_episodes

    for episode in range(num_episodes):
        print(f"--- Episode: {episode + 1} ---")
        # 重置环境，获取初始观察值
        observation = env.reset()
        num_steps = 50 if short_exec_steps is None else short_exec_steps
        floorplan_img_path = os.path.join(scene_config["scene_path"], scene_config["scene_id"], 'floorplan.png')
        floorplan_img = cv2.imread(floorplan_img_path)
        for step in range(num_steps):
            if env.action_space: 
                action = env.action_space.sample()
            else:
                action = None 
            # take action
            state, reward, done, info = env.step(action)
            
            # get observation
            robot_pos = env.robots[0].get_position() # ground truth
            robot_ori = env.robots[0].get_rpy()[2] # ground truth
            robot_ori = [np.sin(robot_ori), np.cos(robot_ori)]
            target_pos = env.task.target_pos
            obs_img = state["rgb"]
            agent.update_vision_input(
                obs_img=obs_img,
                floorplan_img=floorplan_img,
                obs_pos=robot_pos,
                goal_pos=target_pos,
                obs_ori=robot_ori
            )
            if done:
                print(f"Episode {episode + 1} 在 {step + 1} 步后结束。")
                break
        
        if not done and num_episodes > 1: # 避免在只运行少数步骤时打印
                print(f"Episode {episode + 1} 达到最大步数 ({num_steps})。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # 设置日志级别
    run_headless = False        # True: 无GUI运行, False: 带GUI运行
    run_episodes = 3            # 运行多少个 episode
    run_steps_per_episode = 200 # 每个 episode 运行多少步

    
    model_path = "checkpoints\ema_0.pth" # 模型路径
    scene_config_path = "test/load_igibson_scene.yaml"
    model_config_path = 'flona.yaml'
    model_config = None
    
    main(headless=run_headless, short_exec_episodes=run_episodes, short_exec_steps=run_steps_per_episode,scene_config_path=scene_config_path, model_config_path=model_config_path)