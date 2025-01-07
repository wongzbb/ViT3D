import os
import random
import shutil

def move_random_npy_files(source_dir, destination_dir, num_files=300):
    """
    从 source_dir 中随机选择 num_files 个 .npy 文件并移动到 destination_dir。

    :param source_dir: 源文件夹路径
    :param destination_dir: 目标文件夹路径
    :param num_files: 需要随机选择并移动的文件数量
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    all_npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    if len(all_npy_files) < num_files:
        raise ValueError(f"源文件夹中仅有 {len(all_npy_files)} 个 .npy 文件，不足以选出 {num_files} 个。")
    
    selected_files = random.sample(all_npy_files, num_files)
    
    for file_name in selected_files:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(destination_dir, file_name)
        shutil.move(src_path, dst_path)
        print(f"移动文件: {src_path} -> {dst_path}")
    
    print(f"已成功移动 {num_files} 个 .npy 文件到 {destination_dir}")

if __name__ == "__main__":
    source_folder = "/root/code/MRIclass/datasets/train"       
    destination_folder = "/root/code/MRIclass/datasets/test"  
    
    # 随机选择并移动 300 个 .npy 文件
    move_random_npy_files(source_folder, destination_folder, num_files=385)
