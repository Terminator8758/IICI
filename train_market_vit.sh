# Market1501, train with ViT-S backbone: 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vit.py --dataset_name 'Market1501' --data_dir '/path/to/dataset/' --logs_dir 'Market_logs/' --mcnl_negK 20 --split_subcamera 'False' --has_aug_transform 'False'
