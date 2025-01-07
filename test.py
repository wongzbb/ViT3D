import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from glob import glob
from time import time
import argparse
from loguru import logger
import os
from omegaconf import OmegaConf
from tools import find_model_ema, find_model_model, cleanup, create_logger
from models import ViT3D_models
from datasets.data_loader import get_sampler, NpyDataset, Resize4D
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
import torch.nn.functional as F

def main(config):
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert config.sample_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.sample_global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    logger = create_logger(None)

    # Create model:
    model = ViT3D_models[config.model](
        input_size=config.img_size,
        in_channels=1,
        num_frames=config.num_frames,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=2,
        learn_sigma=True,
        attention_mode='math',  #xformers flash math
    )

    # load ckpt
    if config.use_ema:
        model_state_dict_ = find_model_ema(config.sample_ckpt_path)
    else:  
        model_state_dict_ = find_model_model(config.sample_ckpt_path)
    model.load_state_dict(model_state_dict_)

    # log
    logger.info(f"Loaded model from {config.sample_ckpt_path}")

    model.eval()
    model = DDP(model.to(device), device_ids=[rank])

 
    transform = Resize4D(out_size=(config.img_size, config.img_size))
    test_dataset = NpyDataset(config.test_npy_dir, config.excel_path, transform=transform, num_frames=config.num_frames)

    sampler=get_sampler(test_dataset)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=int(config.sample_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=False
    ) # When using a DistributedSampler, you should set shuffle to False.
    if rank == 0:
        logger.info(f"Dataset contains {len(test_dataset)}.")

    model.eval()  # important! This enables embedding dropout for classifier-free guidance


    auroc_metric = MulticlassAUROC(num_classes=2, average='none').to(device)
    accuracy_metric = MulticlassAccuracy(num_classes=2, average='weighted').to(device)
    item = 0
    for slice3D_data in test_loader:
        item += 1
        if item == 30:
            break
        x = slice3D_data[0].to(device, non_blocking=True)
        y = slice3D_data[1].to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(x)

        # logits = F.softmax(logits, dim=-1)

        # preds_for_pos = F.softmax(logits, dim=-1)[:, 1]
        auroc_metric.update(logits, y)
        accuracy_metric.update(logits, y)

        logger.info(f"item: {item}, logits: {logits}, Labels: {y}")

    auroc_score = auroc_metric.compute()
    accuracy_score = accuracy_metric.compute()

    print(f"AUROC: {auroc_score}")
    print(f"Accuracy: {accuracy_score}")

    logger.info("Done!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)





