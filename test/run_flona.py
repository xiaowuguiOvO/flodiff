import os
import time
from igibson.envs.igibson_env import iGibsonEnv
import logging
import FloNaAgent
import training.train_utils as train_utils
def main(headless=False, short_exec_episodes=3, short_exec_steps=200):
    config_file = "test/load_igibson_scene.yaml"

    print(f"将要加载的配置文件: {config_file}")
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 未找到。请确保它与脚本在同一目录，或者提供正确路径。")
        return

    env_mode = "headless" if headless else "gui_interactive"

    print(f"以 '{env_mode}' 模式启动环境...")
    env = iGibsonEnv(
        config_file=config_file,
        mode=env_mode
    )

    try:
        num_episodes = 1 if short_exec_episodes is None else short_exec_episodes

        for episode in range(num_episodes):
            print(f"--- Episode: {episode + 1} ---")
            # 重置环境，获取初始观察值
            observation = env.reset()
            
            num_steps = 50 if short_exec_steps is None else short_exec_steps

            for step in range(num_steps):
                if env.action_space: 
                    action = env.action_space.sample()
                else:
                    action = None 

                # 执行动作
                state, reward, done, info = env.step(action)

                if not headless:
                    time.sleep(0.01) # 调整这个值来改变模拟速度

                if done:
                    print(f"Episode {episode + 1} 在 {step + 1} 步后结束。")
                    break
            
            if not done and num_episodes > 1: # 避免在只运行少数步骤时打印
                 print(f"Episode {episode + 1} 达到最大步数 ({num_steps})。")


    except Exception as e:
        print(f"在模拟过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("正在关闭环境...")
        env.close() # 非常重要：确保在结束时关闭环境以释放资源

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # 设置日志级别
    run_headless = False        # True: 无GUI运行, False: 带GUI运行
    run_episodes = 3            # 运行多少个 episode
    run_steps_per_episode = 200 # 每个 episode 运行多少步

    model_path = "checkpoints\ema_0.pth" # 模型路径
    
    try:
        agent = FloNaAgent.FloNaAgent(model_path=model_path)
        print('suceess load model')
    except Exception as e:
        print(f"error load model: {e}")

    # main(headless=run_headless, short_exec_episodes=run_episodes, short_exec_steps=run_steps_per_episode)