## Pytorch implementation of [TokenMix (ECCV 2022)](https://arxiv.org/abs/2207.08409)

![tenser](assets/tokenmix.png)

This repo is the offcial implementation of the paper [TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers](https://arxiv.org/abs/2207.08409)

```
@article{UniNet,
  author  = {Jihao Liu, Boxiao Liu, Hang Zhou, Yu Liu, Hongsheng Li},
  journal = {arXiv:2207.08409},
  title   = {TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers},
  year    = {2022},
}
```

### Update
8/9/2022 Update the source code.

### Preparation
#### Data
Following [TokenLabeling](https://github.com/zihangJiang/TokenLabeling) to prepare ImageNet data and label maps generated with [NFNet-F6](https://arxiv.org/abs/2102.06171).
#### Environment
The code is tested with ```torch==1.11``` and ```timm==0.5.4```.

### Run experiments

Currently, we supporting running experiments with slurm.
You can reproduce the results of Deit-small as follows: 

```sh exp/deit_small/run.sh partition```

### Models 

|Model|epochs|Top-1 Acc.|Ckpt|
|-|:-:|:-:|:-:|
|Deit-tiny|300|73.2|-|
|Deit-small| 300 | 80.8|-|
|Deit-base|300|82.9|-|
