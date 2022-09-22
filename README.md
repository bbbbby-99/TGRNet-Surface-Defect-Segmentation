# Code and datasets of Triplet--Graph-Reasoning-Network-for-Few-shot-Metal-Generic-Surface-Defect-Segmentation

## Dataset:

You can contact ours by email to get the dataset-decompression-password（yanqibao1997@gmail.com）

##Config
  Before training you need to modify the *.YAML file path in the config folder, such as:
  ```js
  data_root: '../MSD-Seg2/'
  train_list: './data_list/train/fold_0_defective.txt'
  trainnom_list: './data_list/train/fold_0_clean.txt'
  val_list: './data_list/test/fold_0_defective.txt'
  valnom_list: './data_list/test//fold_0_clean.txt'
  ```
## Training：

You can directly use our dataset for training.

you can use your dataset for training, this process requires regenerating the "data_list" file.


## Citing this paper:

If you use our code or data in your research, please use the following BibTeX entry.
 ```js
@article{bao2021triplet,
  title={Triplet-graph reasoning network for few-shot metal generic surface defect segmentation},
  author={Bao, Yanqi and Song, Kechen and Liu, Jie and Wang, Yanyan and Yan, Yunhui and Yu, Han and Li, Xingjie},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={70},
  pages={1--11},
  year={2021},
  publisher={IEEE}
}
```
