import nibabel as nib

def get_nii_dimensions(file_path):
    """
    加载 .nii 文件并输出其尺寸信息。
    
    :param file_path: .nii 文件的路径
    """
    # 加载 NIfTI 文件
    nii = nib.load(file_path)
    
    # 获取数据的形状
    data_shape = nii.shape  # 通常为 (X, Y, Z) 或 (X, Y, Z, T)

    print(f"数据形状: {data_shape}")  #(528, 528, 187)


get_nii_dimensions('/root/code/MRIclass/datasets/data/CHENGYING_35Y_0002944322.nii')