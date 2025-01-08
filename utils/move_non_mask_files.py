import os
import shutil

def move_non_intersect_files(dirA, dirB, dirC):
    filesA = os.listdir(dirA)
    filesB = os.listdir(dirB)

    filesB_set = set(filesB)

    for filename in filesA:
        if filename not in filesB_set:
            print(f"文件 {filename} 不在目录 {dirB} 中")
            src_path = os.path.join(dirA, filename)
            dst_path = os.path.join(dirC, filename)
            shutil.move(src_path, dst_path)
            print(f"已移动文件: {filename}")

if __name__ == "__main__":
    dirA = "/root/code/MRIclass/datasets/test"
    dirB = "/root/code/MRIclass/datasets/mask_npy"
    dirC = "/root/code/MRIclass/datasets/data_non"

    move_non_intersect_files(dirA, dirB, dirC)
