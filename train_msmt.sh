# train MSMT with resnet50-nonlocal backbone:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset_name 'MSMT17' --data_dir '/path/to/dataset/' --logs_dir 'MSMT_logs/' --mcnl_negK 50 --split_subcamera 'True'

