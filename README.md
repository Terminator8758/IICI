# Learning Intra and Inter-Camera Invariance for Isolated Camera supervised Person Re-Identification

[[Paper]](https://arxiv.org/abs/2311.01155) 

This repository is the implementation of [Learning Intra and Inter-Camera Invariance for Isolated Camera supervised Person Re-Identification, ACM MultiMedia 2023](https://arxiv.org/abs/2311.01155). The proposed method IICI targets at isolated camera supervised re-ID, and achieves state-of-the-art performance on multiple re-ID benchmarks.

<img src="figs/framework.png" style="zoom:50%;" />

## Requirements

### Environment
PyTorch >= 1.8

### Installation

```shell
git clone https://github.com/Terminator8758/IICI.git
cd IICI
```

### Prepare Datasets
Download the re-ID datasets Market-1501, MSMT17. Then put them under a folder such as '/path/to/dataset/'.


## Training

We utilize 4 GPUs for training. Performance reported in the paper can be obtained by running the following commands:

Train on Market-1501 using ResNet-Nonlocal backbone (default):
```shell
bash train_market.sh 
```
Train on Market-1501 using ViT-S backbone:
```shell
bash train_market_vit.sh 
```

Train on MSMT17 using ResNet-Nonlocal backbone (default):
```shell
bash train_msmt.sh 
```

Train on MSMT17 using ViT-S backbone:
```shell
bash train_msmt_vit.sh 
```

## Result
<img src="figs/result.png" style="zoom:45%;" />

## Citation
If you find this code useful for your research, please kindly cite our paper:
```
@article{2023_wang_iici,
    title={Learning Intra and Inter-Camera Invariance for Isolated Camera supervised Person Re-Identification},
    author={Menglin Wang and Xiaojin Gong},
    journal={ACM MultiMedia Conference},
    year={2023}
}
```
