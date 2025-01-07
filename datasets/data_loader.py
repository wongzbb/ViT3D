import os
import re
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchio as tio
import random
from scipy.ndimage import rotate
import torch.nn.functional as F
import math
from collections import defaultdict

n_cpu = os.cpu_count()
global_seed = 0


# def random_flip(data):
#     axes = [2, 3] 
#     for axis in axes:
#         if random.random() > 0.5:
#             data = np.flip(data, axis=axis).copy()
#     return data
# def random_rotate(data, degrees=10):
#     angle = random.uniform(-degrees, degrees)
#     data = rotate(data, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
#     angle = random.uniform(-degrees, degrees)
#     data = rotate(data, angle, axes=(1, 3), reshape=False, order=1, mode='constant', cval=0)
#     return data
# def random_add_noise(data, noise_level=0.1):
#     noise = np.random.randn(*data.shape) * noise_level
#     data = data + noise
#     data = np.clip(data, -1, 1)
#     return data

class Resize4D:
    def __init__(self, out_size=(256, 256)):
        self.out_size = out_size

    def __call__(self, x):
        C, D, H, W = x.shape
        out_H, out_W = self.out_size
        x = x.view(C * D, 1, H, W) 
        x = F.interpolate(x, size=(out_H, out_W), mode='bilinear', align_corners=False)
        x = x.view(C, D, out_H, out_W) 
        return x


class NpyDataset(Dataset):
    def __init__(self, npy_dir, excel_path, transform=None, num_frames=120, oversampled=False):
        self.npy_dir = npy_dir
        self.transform = transform
        self.df = pd.read_excel(excel_path)

        self.id_col = '患者编号'  
        self.label_col_index = -3 

        self.num_frames = num_frames

        self.df['patient_id'] = self.df[self.id_col].astype(str)

        self.id_to_label = {}
        for _, row in self.df.iterrows():
            patient_id = row['patient_id']
            # label = row.iloc[self.label_col_index]
            label = row['SLN状态（0_无转移，1_转移）']
            self.id_to_label[patient_id] = label


        self.npy_files = glob(os.path.join(self.npy_dir, '**', '*.npy'), recursive=True)
        if not self.npy_files:
            raise ValueError(f"No .npy files found in {self.npy_dir}")

        self.data_labels = []
        for file in self.npy_files:
            filename = os.path.basename(file)
            match = filename.split('_')[-1]
            if match == 'right.npy' or match == 'left.npy' or match == 'L.npy' or match == 'R.npy' or match == '1.npy':
                match = filename.split('_')[-2] + '_' + filename.split('_')[-1]

            if match:
                patient_id = match.split('.')[0]
                # print(f"patient_id: {patient_id}")
                label = self.id_to_label.get(patient_id)
                if label is not None:
                    self.data_labels.append((file, label))
                # else:
                #     print(f"文件 {filename} 的 patient_id '{patient_id}' 在 Excel 表中未找到匹配项")
            # else:
            #     print(f"文件名 {filename} 不符合预期格式")

        # if not self.data_labels:
        #     raise ValueError("没有找到任何匹配的 .npy 文件和标签。")

        if oversampled:
            label_counts = defaultdict(int)
            for _, label in self.data_labels:
                label_counts[label] += 1
            num_label_0 = label_counts[0]
            num_label_1 = label_counts[1]

            print(f"num_label_0: {num_label_0}, num_label_1: {num_label_1}")

            oversample_ratio = num_label_0 // num_label_1
            remainder = num_label_0 % num_label_1

            oversampled_data = []
            for file, label in self.data_labels:
                if label == 1:
                    oversampled_data.extend([(file, label)] * (oversample_ratio-1))
                    if remainder > 0:
                        oversampled_data.append((file, label))
                        remainder -= 1

            print(f"oversampled_data: {len(oversampled_data)}")

            self.data_labels.extend(oversampled_data)

            random.shuffle(self.data_labels)

            # new_label_counts = defaultdict(int)
            # for _, label in self.data_labels:
            #     new_label_counts[label] += 1

            # print("过采样后的类别分布:", new_label_counts)



    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):

        file_path, label = self.data_labels[idx]
        data = np.load(file_path)

        # data = self.normalize_data(data)

        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)  #  (1, H, W, D)
        elif data.ndim == 4:
            data = data  

        data = data.transpose(0, 3, 1, 2)  # TorchIO expects (C, D, H, W)

        data = torch.from_numpy(data).float()
        if self.transform:
            data = self.transform(data)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=data)  # TorchIO expects (C, D, H, W)
        )
        
        augmented_data = subject.image.data.squeeze(0).numpy()  # 回到 (D, H, W)
        augmented_data = torch.from_numpy(augmented_data).float()

        # print(f"augmented_data.shape: {augmented_data.shape}")
        augmented_data = augmented_data.unsqueeze(1)
        # print(f"augmented_data.shape: {augmented_data.shape}")

        label = torch.tensor(label).long() 


        return augmented_data[:self.num_frames,:,:,:], label

    def normalize_data(self, data):
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data)
        normalized = (data - data_min) / (data_max - data_min)  
        normalized = normalized * 2 - 1 
        return normalized

dist.init_process_group("nccl")
rank = dist.get_rank()
def get_sampler(dataset_):
    sampler = DistributedSampler(dataset_, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=global_seed)
    return sampler


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset: Dataset, weights, num_replicas=None, rank=None, replacement=True, seed=0):
        self.dataset = dataset
        self.weights = weights
        self.num_samples = len(dataset) 
        self.replacement = replacement

        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.multinomial(
            self.weights, 
            self.num_samples, 
            self.replacement, 
            generator=g
        ).tolist()

        indices_rank = indices[self.rank::self.num_replicas]
        return iter(indices_rank)

    def __len__(self):
        return math.ceil(self.num_samples / self.num_replicas)

def create_data_loader(npy_dir, excel_path, batch_size=32, shuffle=True, num_workers=4):
    transform = Resize4D(out_size=(256, 256))
    dataset = NpyDataset(npy_dir, excel_path, transform=transform, oversampled=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

if __name__ == "__main__":
    source_directory = '/root/code/MRIclass/datasets/train/'   
    excel_file_path = '/root/code/MRIclass/datasets/SLN_WCH_final version_new.xlsx'   

    data_loader = create_data_loader(source_directory, excel_file_path, batch_size=1, shuffle=True, num_workers=4)
    print(f"数据集大小: {len(data_loader.dataset)}")

    ii = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        print(f"批次 {batch_idx+1}:")
        print(f" - 数据形状: {data.shape}")
        print(f" - 标签: {labels}")
        if labels[0] == 1:
            ii += 1
    print(ii)
        # 这里可以添加您的训练代码
        # 例如：
        # outputs = model(data)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        # if batch_idx >= 2:  # 仅打印前3个批次
        #     break
