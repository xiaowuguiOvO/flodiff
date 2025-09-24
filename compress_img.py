# 文件名: preprocess_images_v2.py
import os
import shutil
from PIL import Image
from tqdm import tqdm
import sys

# --- 1. 配置参数 ---

# 源数据根目录 (包含 train 和 test)
SOURCE_ROOT = "datasets/scenes_117"

# 目标数据根目录 (脚本会自动创建 train_96 和 test_96)
TARGET_ROOT = "." 

# 目标图片尺寸
IMAGE_SIZE = (96, 96)

# 需要处理的数据集划分
SPLITS = ["train", "test"]

# --- 2. 主处理逻辑 ---

def process_all_files():
    """
    遍历源目录中的所有文件和文件夹。
    - 缩放 .png 图片并保存到目标目录。
    - 直接复制所有其他文件到目标目录。
    """
    # 检查源数据目录是否存在
    if not os.path.isdir(SOURCE_ROOT):
        print(f"错误：源数据目录 '{SOURCE_ROOT}' 不存在。")
        print("请确保此脚本与 'scenes_117' 目录在同一文件夹下运行。")
        sys.exit(1)

    # 对 train 和 test 分别进行处理
    for split in SPLITS:
        source_split_dir = os.path.join(SOURCE_ROOT, split)
        target_split_dir = os.path.join(TARGET_ROOT, f"{split}_96")

        if not os.path.isdir(source_split_dir):
            print(f"警告：找不到源目录 '{source_split_dir}'，将跳过。")
            continue

        print(f"--- 开始处理 '{split}' 数据集 ---")
        print(f"源目录: {source_split_dir}")
        print(f"目标目录: {target_split_dir}")

        # 步骤 1: 收集所有需要处理的文件路径
        print("正在收集所有文件路径...")
        all_source_files = []
        for root, _, files in os.walk(source_split_dir):
            for file in files:
                all_source_files.append(os.path.join(root, file))
        
        if not all_source_files:
            print("在源目录中没有找到任何文件。")
            continue

        print(f"共找到 {len(all_source_files)} 个文件。开始处理...")

        # 步骤 2: 遍历所有文件，进行处理
        for source_path in tqdm(all_source_files, desc=f"处理 {split} 文件"):
            try:
                # 计算目标路径
                target_path = source_path.replace(source_split_dir, target_split_dir, 1)
                
                # 获取目标文件所在的目录并创建
                target_folder = os.path.dirname(target_path)
                os.makedirs(target_folder, exist_ok=True)

                # 判断文件类型并执行相应操作
                if source_path.lower().endswith('.png'):
                    # --- 处理图片：缩放并保存 ---
                    with Image.open(source_path) as img:
                        img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        img_resized.save(target_path)
                else:
                    # --- 处理其他文件：直接复制 ---
                    shutil.copy2(source_path, target_path)
            
            except Exception as e:
                print(f"\n处理文件 {source_path} 时出错: {e}")

        print(f"--- '{split}' 数据集处理完成！--- \n")

if __name__ == "__main__":
    process_all_files()
    print("所有处理任务已完成。")
    print(f"处理后的数据已保存到 '{os.path.join(TARGET_ROOT, 'train_96')}' 和 '{os.path.join(TARGET_ROOT, 'test_96')}' 目录中。")