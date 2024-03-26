# Market1501, train with ViT-S backbone: 
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_vit.py --dataset_name 'Market1501' --data_dir '/home1/wml/dataset/' --logs_dir 'Market_logs/' --mcnl_negK 20 --split_subcamera 'False' --has_aug_transform 'False'



# MSMT17:
# train with resnet50-nonlocal backbone:
#CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name 'MSMT17' --data_dir '/data/wml/dataset/' --logs_dir 'MSMT_logs/' --mcnl_negK 50  --split_subcamera True  
