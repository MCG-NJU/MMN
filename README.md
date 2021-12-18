# [AAAI 2022] Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding
> Official Pytorch implementation of Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding (AAAI 2022).
>
> Paper is at https://arxiv.org/pdf/2109.04872.pdf.
>
> Paper explanation in Zhihu (in Chinese) is at https://zhuanlan.zhihu.com/p/446203594.

### Abstract

Temporal grounding aims to localize a video moment which is semantically aligned with a given natural language query. Existing methods typically apply a detection or regression pipeline on the fused representation with the research focus on designing complicated prediction heads or fusion strategies. Instead, from a perspective on temporal grounding as a metric-learning problem, we present a Mutual Matching Network (MMN), to directly model the similarity between language queries and video moments in a joint embedding space. This new metric-learning framework enables fully exploiting negative samples from two new aspects: constructing negative cross-modal pairs in a mutual matching scheme and mining negative pairs across different videos. These new negative samples could enhance the joint representation learning of two modalities via cross-modal mutual matching to maximize their mutual information. Experiments show that our MMN achieves highly competitive performance compared with the state-of-the-art methods on four video grounding benchmarks. Based on MMN, we present a winner solution for the HC-STVG challenge of the 3rd PIC workshop. This suggests that metric learning is still a promising method for temporal grounding via capturing the essential cross-modal correlation in a joint embedding space.

### Updates

Dec, 2021 - We uploaded the code and trained weights for Charades-STA, ActivityNet-Captions and TACoS datasets.

Todo: The code for spatio-temporal video grounding (HC-STVG dataset) will be available soon.

### Datasets

* Download the [video feature](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)  provided by [2D-TAN](https://github.com/microsoft/2D-TAN).  The groundtruth file is already uploaded in the `dataset` folder, where I directly use the [groundtruth file of 2D-TAN](https://github.com/microsoft/2D-TAN/tree/master/data) for ActivityNet and TACoS dataset, and I change the original form of Charades dataset in [2D-TAN (as .txt file)](https://github.com/microsoft/2D-TAN/tree/master/data/Charades-STA/) to be the same form with other two datasets (as .json file) for more simplicity of my code for loading datasets.
* Extract and put the feature in the corresponding dataset in the  `dataset` folder. For configurations of feature/groundtruth's paths, please refer to `./mmn/config/paths_catalog.py`. (ann_file is the annotation, feat_file is the video feature)

### Dependencies

Our code is developed on the [third-party implementation of 2D-TAN](https://github.com/ChenJoya/2dtan), so we have similar dependencies with it, such as:

```
yacs h5py terminaltables tqdm pytorch transformers 
```

### Quick Start

We provide scripts for simplifying training and inference. For training our model, we provide a script for each dataset (e.g.,` ./scripts/tacos_train.sh`). For evaluating the performance, we provide `./scripts/eval.sh`. 

For example, for training model in TACoS dataset in `tacos_train.sh`, we need to select the right config in `config` and decide the GPU by yourself in `gpus` (gpu id in your server) and `gpun` (total number of gpus).

```
# find all configs in configs/
config=pool_tacos_128x128_k5l8
# set your gpu id
gpus=0,1
# number of gpus
gpun=2
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.3
master_port=29511
```

Similarly, to evaluate the model, just change the information in `eval.sh`. Our trained weights for three datasets are in the [Google Drive](https://drive.google.com/drive/folders/11zD3YAUSh3u7HPw0eWSJG2cIiMpj_P2K?usp=sharing).

### Citation

If you find our code useful, please generously cite our paper. (AAAI version bibtex will be updated later)

```
@article{DBLP:journals/corr/abs-2109-04872,
  author    = {Zhenzhi Wang and
               Limin Wang and
               Tao Wu and
               Tianhao Li and
               Gangshan Wu},
  title     = {Negative Sample Matters: {A} Renaissance of Metric Learning for Temporal
               Grounding},
  journal   = {CoRR},
  volume    = {abs/2109.04872},
  year      = {2021}
}
```

### Contact

For any question, please raise an issue (preferred) or contact

```
Zhenzhi Wang: zhenzhiwang@outlook.com
```
### Acknowledgement

We appreciate [2D-TAN](https://github.com/microsoft/2D-TAN) for video feature and configurations, and the [third-party implementation of 2D-TAN](https://github.com/ChenJoya/2dtan) for its implementation with `DistributedDataParallel`. Disclaimer: the performance gain of this [third-party implementation](https://github.com/ChenJoya/2dtan) is due to a tiny mistake of adding val set into training, yet our reproduced result is similar to the reported result in 2D-TAN paper.

