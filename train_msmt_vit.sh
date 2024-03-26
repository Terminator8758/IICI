# MSMT, train with ViT-S backbone: 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vit.py --dataset_name 'MSMT17' --data_dir '/path/to/dataset/' --logs_dir 'MSMT_logs/' --mcnl_negK 50 --split_subcamera 'True' --has_aug_transform 'False'


