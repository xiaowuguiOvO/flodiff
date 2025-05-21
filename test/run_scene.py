import os
import time
from igibson.envs.igibson_env import iGibsonEnv
import logging
def main(headless=False, short_exec_episodes=3, short_exec_steps=200):
    """
    加载并运行一个由YAML文件配置的iGibson环境。
    YAML文件中的 scene_id 可以被修改以加载不同的场景。
    """
    # 假设配置文件与此脚本在同一目录下
    config_file = "test/load_igibson_scene.yaml"

    print(f"将要加载的配置文件: {config_file}")
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 未找到。请确保它与脚本在同一目录，或者提供正确路径。")
        return

    # 创建 iGibsonEnv 环境
    # mode: "gui_interactive" - 带GUI且可交互
    #       "gui_non_interactive" - 带GUI但不可通过鼠标键盘控制相机/机器人
    #       "headless" - 无GUI，后台运行
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
            # 对于 "dummy" 任务，观察值可能很简单
            observation = env.reset()
            
            # observation 是一个字典，包含了YAML中定义的 'output' 数据
            # 例如: observation['rgb'], observation['depth'], observation['scan']

            num_steps = 50 if short_exec_steps is None else short_exec_steps

            for step in range(num_steps):
                # 获取一个随机动作 (如果机器人和任务支持)
                if env.action_space: # 检查是否有可用的动作空间
                    action = env.action_space.sample()
                else:
                    action = None # 如果没有动作空间，则不执行动作

                # 执行动作
                state, reward, done, info = env.step(action)

                # 如果不是无头模式，可以稍微暂停一下方便观察
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
    
    # --- 在这里配置 ---
    run_headless = False        # True: 无GUI运行, False: 带GUI运行
    # 如果想快速测试，可以减少 episode 和 step 数量
    # run_episodes = 1
    # run_steps_per_episode = 50
    run_episodes = 3            # 运行多少个 episode
    run_steps_per_episode = 200 # 每个 episode 运行多少步
    # --- 配置结束 ---

    main(headless=run_headless, short_exec_episodes=run_episodes, short_exec_steps=run_steps_per_episode)