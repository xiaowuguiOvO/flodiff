# inspect_npy.py
import numpy as np
import os

# --- 你要检查的文件路径 ---
# 使用 os.path.join 来确保路径在不同操作系统下都能正常工作
file_path = os.path.join('datasets', 'scenes_117', 'test', 'Eudora_0', 'traj_163', 'traj_163.npy')
# -------------------------

try:
    # 设置打印选项，防止因为数组太大而省略内容，并设置精度
    np.set_printoptions(threshold=np.inf, suppress=True, precision=4)
    
    # 从 .npy 文件加载数据
    data = np.load(file_path)

    print(f"成功加载文件: {file_path}\n")

    # 打印数组的形状 (dimensions)
    print(f"数组形状 (Shape): {data.shape}")

    # 打印数组的数据类型
    print(f"数据类型 (dtype): {data.dtype}\n")

    # 打印数组的全部内容
    print("--- 文件内容 ---")
    print(data)
    print("--- 内容结束 ---\n")

except FileNotFoundError:
    print(f"[错误] 文件未找到: {file_path}")
    print("请确保你是在正确的项目根目录下运行此脚本。")
except Exception as e:
    print(f"发生了未知错误: {e}")