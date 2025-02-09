import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import wandb
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.functional.regression import pearson_corrcoef

from torch.cuda.amp import GradScaler, autocast
from einops import rearrange
from omegaconf import OmegaConf

from tools import update_ema, find_model_ema, find_model_model, requires_grad, cleanup, create_logger, clip_grad_norm_, load_dino_model

from models import ViT3D_models

from datasets.data_loader import get_sampler, NpyDataset, Resize4D, DistributedWeightedSampler


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(config):
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    assert config.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{config.results_dir}/*"))
        model_string_name = config.model.replace("/", "-")  

        experiment_dir = f"{config.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)

        if config.wandb:
            wandb.init(project='3D_'+config.model.replace('/','_'))
            wandb.config = {"learning_rate": config.lr, 
                            "epochs": config.epochs, 
                            "batch_size": config.global_batch_size,
                            "dt-rank": config.dt_rank,
                            "autocast": config.autocast,
                            "margin": config.margin,
                            "save-path": experiment_dir,
                            "autocast": config.autocast,
                            }

        logger.info(f"Experiment directory created at {experiment_dir}")
        OmegaConf.save(config, os.path.join(experiment_dir, 'config.yaml'))
    else:
        logger = create_logger(None)

     # Create model:
    if config.use_DINO:
        model = ViT3D_models[config.model](
            input_size=config.img_size,
            in_channels=1,
            num_frames=config.num_frames//2,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=2,
            learn_sigma=True,
            attention_mode='math',  #xformers flash math
        )
    else:
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

    if config.init_from_pretrain_ckpt:
        #load model
        model_state_dict_ = find_model_model(config.pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        #load ema
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model_ema(config.pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        # log
        logger.info(f"Loaded pretrain model from {config.pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])

    if config.use_compiler:
        model = torch.compile(model)

    if rank == 0:
        logger.info(f"ViT3D Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Use half-precision training? {config.autocast}")

    
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # load DINO model
    if config.use_DINO:
        dino = load_dino_model(device=device, pretrained_path=config.DINO_ckpt_path)


    
    transform = Resize4D(out_size=(config.img_size, config.img_size))

    train_dataset = NpyDataset(config.train_npy_dir, config.mask_dir, config.excel_path, transform=transform, num_frames=config.num_frames, oversampled=config.oversampled, use_mask=config.use_mask)

    sampler=get_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(config.global_batch_size // dist.get_world_size()), 
        shuffle=False, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True
    ) # When using a DistributedSampler, you should set shuffle to False.

    if rank == 0:
        logger.info(f"Dataset contains {len(train_dataset)}.")


    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if config.init_from_pretrain_ckpt:
        train_steps = config.init_train_steps
    else:
        train_steps = 0

    log_steps = 0
    running_loss = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Training for {config.epochs} epochs...")

    # class_weights = torch.tensor([1.0, 4.0]).to(device)

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")

        item = 0
        for (x, y) in train_loader:
            item+=1
            x = x.to(device)
            y = y.to(device)
            # mask = slice3D_data['mask'].to(device, non_blocking=True)
            # print(x.shape)

            # print(x.shape)
            if config.use_DINO:
                mask = x[:,config.num_frames//2:,:,:,:]
                x = x[:,:config.num_frames//2,:,:,:]
                

            if config.autocast:
                with autocast():
                    logits, layer_outputs = model(x)
            else:
                logits, layer_outputs = model(x)

            if config.use_focal_loss:
                y_one_hot = F.one_hot(y, num_classes=2).float()
                total_loss = sigmoid_focal_loss(logits, y_one_hot, alpha=0.25, gamma=2, reduction='mean')
            else:
                total_loss = F.cross_entropy(logits, y)

            if config.use_DINO:
                with torch.no_grad(): 
                    attentions = dino.get_special_layers(rearrange(mask.repeat(1, 1, 3, 1, 1), 'b f c h w -> (b f) c h w').contiguous(), config.DINO_selected_layers)
                    attentions = [item[:, 1:, :] for item in attentions]

                DINO_loss = 0
                k = 0
                for i in config.DINO_selected_layers:
                    layer_outputs_ = rearrange(layer_outputs[i], '(b f) l d -> b (f l d)', b = config.global_batch_size)
                    attentions_ = rearrange(attentions[k], '(b f) l d -> b (f l d)', b = config.global_batch_size)

                    pearson_loss_ = 0
                    for j in range(config.global_batch_size):
                        pearson_loss_ = 1 - pearson_corrcoef(layer_outputs_[j], attentions_[j])
                    pearson_loss_ = pearson_loss_ / config.global_batch_size
                    DINO_loss += pearson_loss_
                    k += 1

                # print(f"DINO_loss: {DINO_loss}")
                total_loss += config.DINO_loss_weight * DINO_loss
                    


            # total_loss = sigmoid_focal_loss(logits, y_one_hot, alpha=0.75, gamma=10, reduction='mean')

            if rank == 0 and config.wandb:
                wandb.log({"loss": total_loss.item()})

            if config.autocast:
                with autocast():
                    scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if config.use_clip_norm and config.start_clip_epoch <= epoch:
                gradient_norm = clip_grad_norm_(model.module.parameters(), config.clip_max_norm, clip_grad=True)
            else:
                gradient_norm = 1.0

            if train_steps % config.accumulation_steps == 0:
                if config.autocast:
                    with autocast():
                        scaler.step(opt)
                        scaler.update()
                        update_ema(ema, model.module)
                        opt.zero_grad()
                else:
                    opt.step()
                    update_ema(ema, model.module)
                    opt.zero_grad()

            # Log loss values:
            running_loss += total_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % config.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                epoch_isfinish = int(config.global_batch_size // dist.get_world_size()) * item / len(train_dataset) * 100
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                if rank == 0:
                    if config.use_DINO:
                        logger.info(f"({epoch_isfinish:.1f}%) (step={train_steps:07d}) Train Loss: {avg_loss:.8f}, DINO_loss: {DINO_loss:.8f},  Train Steps/Sec: {steps_per_sec:.2f}, logits: {logits}, Labels: {y}")
                    else:
                        logger.info(f"({epoch_isfinish:.1f}%) (step={train_steps:07d}) Train Loss: {avg_loss:.8f}, Train Steps/Sec: {steps_per_sec:.2f}, logits: {logits}, Labels: {y}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % config.ckpt_every == 0 and train_steps > 0:
                # # test
                # for slice3D_data in train_loader:
                #     x1 = slice3D_data[0].to(device, non_blocking=True)
                #     y1 = slice3D_data[1].to(device, non_blocking=True)
                #     logits = model(x1)
                #     print(f"test logits: {logits}, Labels: {y1}")
                #     del logits
                #     torch.cuda.empty_cache()

                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": config
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if rank == 0:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    if rank == 0 and config.wandb:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))


                




