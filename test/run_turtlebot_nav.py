import os
import igibson # 导入igibson主库，用于获取其安装路径
from igibson.envs.igibson_env import iGibsonEnv
import time # 用于添加延迟，方便观察

def run_igibson_with_config(config_file_path, headless=False, short_exec=False):
    """
    使用指定的YAML配置文件加载并运行iGibson环境。
    """
    print(f"尝试加载配置文件: {config_file_path}")
    if not os.path.exists(config_file_path):
        print(f"错误: 配置文件 {config_file_path} 未找到!")
        print("请确保路径正确，或者文件存在于指定位置。")
        return

    env = iGibsonEnv(
        config_file=config_file_path,
        mode="headless" if headless else "gui_interactive", # gui_interactive, gui_non_interactive, headless
        # action_timestep=1.0/10.0, # (可选) 动作步长时间
        # physics_timestep=1.0/240.0 # (可选) 物理模拟步长时间
    )

    try:
        print("环境已加载。开始模拟...")
        num_episodes = 1 if short_exec else 3 # 短时执行则1个episode，否则3个

        for episode in range(num_episodes):
            print(f"--- Episode: {episode + 1} ---")
            observation = env.reset() # 重置环境到初始状态，获取初始观察

            # observation 是一个字典，包含了YAML中定义的 'output' 数据
            # 例如: observation['rgb'], observation['depth'], observation['scan'], observation['task_obs']

            num_steps_per_episode = 50 if short_exec else 200 # 每个episode的步数

            for step in range(num_steps_per_episode):
                # 获取一个随机动作 (Turtlebot通常是2D的 [线速度, 角速度])
                action = env.action_space.sample()

                # 执行动作
                state, reward, done, info = env.step(action)

                # print(f"Step: {step + 1}, Reward: {reward:.3f}, Done: {done}")

                if not headless:
                    time.sleep(0.01) # 在GUI模式下稍微暂停，方便观察

                if done:
                    print(f"Episode {episode + 1} 在 {step + 1} 步后结束。")
                    break

            if not done and not short_exec:
                print(f"Episode {episode + 1} 达到最大步数。")

    except Exception as e:
        print(f"在模拟过程中发生错误: {e}")
    finally:
        print("正在关闭环境...")
        env.close() # 非常重要：确保在结束时关闭环境以释放资源

if __name__ == "__main__":
    # --- 配置区 ---
    # 方式1: 假设 turtlebot_nav.yaml 与你的脚本在同一目录下
    # my_config_file = "turtlebot_nav.yaml"

    # 方式2: 使用 iGibson 安装目录中的配置文件 (推荐，如果不想复制文件)
    # 首先找到 iGibson 的安装路径
    try:
        igibson_installation_path = os.path.dirname(igibson.__file__)
        default_config_file = os.path.join(igibson_installation_path, "configs", "turtlebot_static_nav.yaml")
    except ImportError:
        print("错误: iGibson库未找到。请确保已正确安装iGibson。")
        exit()

    # 选择要使用的配置文件路径
    config_to_use = default_config_file # 或者 my_config_file (如果你自己保存了)

    run_igibson_with_config(
        config_file_path=config_to_use,
        headless=False,      # 设置为 True 以无头模式运行 (无GUI)
        short_exec=False     # 设置为 True 以快速运行少量步骤/episode
    )