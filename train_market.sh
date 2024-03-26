# Market1501:
# train with resnet50-nonlocal backbone:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset_name 'Market1501' --data_dir '/path/to/dataset/' --logs_dir 'Market_logs/' --mcnl_negK 20 --split_subcamera 'False'

