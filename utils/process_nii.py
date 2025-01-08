import os
import nibabel as nib
import numpy as np
from glob import glob

def find_min_dimensions(source_dir):
    """
    遍历源目录下所有 .nii 和 .nii.gz 文件，找到最小的宽度、高度和切片数量。

    :param source_dir: 源文件夹路径
    :return: (min_width, min_height, min_slices)
    """
    # 查找所有 .nii 和 .nii.gz 文件，递归遍历子目录
    nii_files = glob(os.path.join(source_dir, '**', '*.nii*'), recursive=True)
    if not nii_files:
        raise ValueError(f"No .nii or .nii.gz files found in {source_dir}")

    min_width = min_height = min_slices = None

    qqq = 0
    for file in nii_files:
        try:
            img = nib.load(file)
            data_shape = img.shape
            if len(data_shape) < 3:
                print(f"文件 {file} 的维度小于3，跳过。")
                continue
            width, height, slices = data_shape[:3]
            if slices < 120 or width < 500 or height < 500 or width!=height:
                qqq += 1
                continue


            if min_width is None or width < min_width:
                min_width = width
                print(f"min_width: {min_width}")
            if min_height is None or height < min_height:
                min_height = height
                print(f"min_height: {min_height}")
            if min_slices is None or slices < min_slices:
                min_slices = slices
                print(f"min_slices: {min_slices}")
        except Exception as e:
            print(f"无法处理文件 {file}: {e}")

    print(f"qqq: {qqq}")
    if min_width is None or min_height is None or min_slices is None:
        raise ValueError("未能找到有效的 NIfTI 文件或文件维度不足。")

    print(f"最小宽度: {min_width}, 最小高度: {min_height}, 最小切片数: {min_slices}")
    # return min_width, min_height, min_slices
    return 500, 500, 120

def center_crop(data, target_width, target_height, target_slices):
    """
    对3D数据进行中心裁剪。

    :param data: 3D numpy数组
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :param target_slices: 目标切片数
    :return: 裁剪后的3D numpy数组
    """
    width, height, slices = data.shape[:3]

    # 计算裁剪起始点
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2
    start_z = (slices - target_slices) // 2

    end_x = start_x + target_width
    end_y = start_y + target_height
    end_z = start_z + target_slices

    # 执行裁剪
    cropped_data = data[start_x:end_x, start_y:end_y, start_z:end_z]

    return cropped_data

def normalize_data(data):
    """
    将数据归一化到[-1, 1]范围内。

    :param data: numpy数组
    :return: 归一化后的numpy数组
    """
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        # 避免除以零，如果所有值相同，则返回全零数组
        return np.zeros_like(data)
    normalized = (data - data_min) / (data_max - data_min)  # 缩放到[0, 1]
    normalized = normalized * 2 - 1  # 缩放到[-1, 1]
    return normalized

def process_nii_files(source_dir, destination_dir, min_width, min_height, min_slices):
    """
    遍历源目录下所有 .nii 和 .nii.gz 文件，进行中心裁剪、归一化并保存为 .npy 文件到目标目录。

    :param source_dir: 源文件夹路径
    :param destination_dir: 目标文件夹路径
    :param min_width: 最小宽度
    :param min_height: 最小高度
    :param min_slices: 最小切片数
    """
    nii_files = glob(os.path.join(source_dir, '**', '*.nii*'), recursive=True)
    if not nii_files:
        raise ValueError(f"No .nii or .nii.gz files found in {source_dir}")

    os.makedirs(destination_dir, exist_ok=True)

    for file in nii_files:
        try:
            img = nib.load(file)
            data = img.get_fdata()
            original_shape = data.shape[:3]

            # 检查是否需要裁剪
            if (original_shape[0] < min_width or
                original_shape[1] < min_height or
                original_shape[2] < min_slices):
                print(f"文件 {file} 的尺寸小于最小尺寸，跳过。")
                continue

            # 中心裁剪
            cropped_data = center_crop(data, min_width, min_height, min_slices)

            # 归一化到[-1, 1]
            normalized_data = normalize_data(cropped_data)

            # 构建目标文件路径
            relative_path = os.path.relpath(file, source_dir)
            # 修改扩展名为 .npy
            base_filename = os.path.splitext(relative_path)[0]
            if base_filename.endswith('.nii'):
                base_filename = os.path.splitext(base_filename)[0]
            target_file = os.path.join(destination_dir, base_filename + '.npy')
            target_dir = os.path.dirname(target_file)
            os.makedirs(target_dir, exist_ok=True)

            # 保存为 .npy 文件
            np.save(target_file, normalized_data)
            print(f"已裁剪、归一化并保存: {target_file}")

        except Exception as e:
            print(f"无法处理文件 {file}: {e}")

def main():
    # 设置源文件夹和目标文件夹路径
    source_directory = '/root/code/MRIclass/datasets/mask_/'          # 替换为您的源文件夹路径
    destination_directory = '/root/code/MRIclass/datasets/mask_npy/'# 替换为您的目标文件夹路径

    # 步骤1：找到最小尺寸
    min_width, min_height, min_slices = find_min_dimensions(source_directory)

    # 步骤2：处理并裁剪所有文件
    process_nii_files(source_directory, destination_directory, min_width, min_height, min_slices)

    print("所有文件已成功处理。")

if __name__ == "__main__":
    main()
