# DivAug: Plug-in Automated Data Augmentation with Explicit Diversity Maximization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official project repository for [DivAug: Plug-in Automated Data Augmentation with Explicit Diversity Maximization](https://arxiv.org/abs/2103.14545) [ICCV 2021].

Zirui Liu*, Haifeng Jin*, Ting-Hsiang Wang, Kaixiong Zhou, and Xia Hu.

**TL; DR.**
DivAug is a unsupervised automated data augmentation method without requiring a separate search process.


## Implementation

The `./cpp_extension` directory contains the C++ implementation of the parallel KMEAN++ algorithm based on OpenMP. 
Our implementation of KMEAN++ is based on [KMEAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) module in scikit-learn.
We also provide the vanilla python implementation of KMEAN++ in `./common/utils:line 48`.

## Install
- Requirements
```
torch>=1.8.0
torchvision>=0.9.0
warmup_scheduler (https://github.com/ildoonet/pytorch-gradual-warmup-lr)
```

- Build
```bash
cd cpp_extension
pip install -v -e .
```
## Reproduce results

See `./scripts`. For example, if you want to train wide-resnet-28-10 with DivAug on CIFAR-100, just run 
```bash
bash scripts/wresnet_28x10divaug_cifar100.sh
```

## Citation

```
@misc{liu2021divaug,
      title={DivAug: Plug-in Automated Data Augmentation with Explicit Diversity Maximization}, 
      author={Zirui Liu and Haifeng Jin and Ting-Hsiang Wang and Kaixiong Zhou and Xia Hu},
      year={2021},
      eprint={2103.14545},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Our codebase is mainly based on 

https://github.com/kakaobrain/fast-autoaugment

https://github.com/ildoonet/unsupervised-data-augmentation

https://github.com/SenWu/dauphin