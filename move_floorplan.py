import os
import shutil

# --- 配置路径 ---
# 源目录：包含所有场景文件夹（如 Eudora_0）的根目录
source_root_dir = "datasets/scenes_117/test"

# 目标目录：要将 floorplan.png 移动到的目标根目录
dest_root_dir = "iGibson/igibson/data/g_dataset"
# -----------------

def move_floorplans():
    """
    自动扫描、解析并移动所有测试场景的 floorplan.png 文件。
    """
    print(f"开始扫描源目录: {source_root_dir}")

    # 检查源目录是否存在
    if not os.path.isdir(source_root_dir):
        print(f"错误：源目录 '{source_root_dir}' 不存在。请检查路径。")
        return

    # 遍历源目录下的所有项目（即 Eudora_0, Hillsdale_1 等）
    for scene_folder_name in os.listdir(source_root_dir):
        full_scene_folder_path = os.path.join(source_root_dir, scene_folder_name)

        # 确保处理的是文件夹
        if os.path.isdir(full_scene_folder_path):
            
            # --- 解析场景基础名 ---
            # 从 'Eudora_0' 这样的名字中提取出 'Eudora'
            try:
                last_underscore_index = scene_folder_name.rfind('_')
                if last_underscore_index == -1:
                    print(f"跳过 '{scene_folder_name}' - 文件夹命名格式不符合预期 (例如 'SceneName_0')。")
                    continue
                base_scene_id = scene_folder_name[:last_underscore_index]
            except Exception as e:
                print(f"解析文件夹名 '{scene_folder_name}' 时出错: {e}。跳过...")
                continue

            # --- 构建源文件和目标文件的完整路径 ---
            source_file_path = os.path.join(full_scene_folder_path, "floorplan.png")
            
            dest_folder_path = os.path.join(dest_root_dir, base_scene_id)
            dest_file_path = os.path.join(dest_folder_path, "floorplan.png")
            
            # --- 执行移动操作 ---
            if os.path.exists(source_file_path):
                # 确保目标文件夹存在，如果不存在则创建
                os.makedirs(dest_folder_path, exist_ok=True)
                
                print(f"正在移动: {source_file_path} -> {dest_file_path}")
                
                # 使用 shutil.move 来移动文件
                shutil.move(source_file_path, dest_file_path)
            else:
                print(f"警告: 在 '{full_scene_folder_path}' 中未找到 floorplan.png，跳过。")

    print("\n所有 floorplan.png 文件处理完毕。")

if __name__ == "__main__":
    move_floorplans()