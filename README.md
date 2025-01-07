## ⏳Train
```
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 train.py --config ./configs/config.yaml
```

## 🎇Test
```
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=12345 --nnodes=1 --nproc_per_node=1 test.py --config ./configs/config.yaml
```