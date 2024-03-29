# CSE-Autoloss

Designing proper loss functions for vision tasks has been a long-standing research direction to advance the capability of existing models. For object detection, the well-established classification and regression loss functions have been carefully designed by considering diverse learning challenges (e.g. class imbalance, hard negative samples, and scale variances). Inspired by the recent progress in network architecture search, it is interesting to explore the possibility of discovering new loss function formulations via directly searching the primitive operation combinations. So that the learned losses not only fit for diverse object detection challenges to alleviate huge human efforts, but also have better alignment with evaluation metric and good mathematical convergence property. Beyond the previous auto-loss works on face recognition and image classification, our work makes the first attempt to discover new loss functions for the challenging object detection from primitive operation levels and finds the searched losses are insightful. We propose an effective convergence-simulation driven evolutionary search algorithm, called CSE-Autoloss, for speeding up the search progress by regularizing the mathematical rationality of loss candidates via two progressive convergence simulation modules: convergence property verification and model optimization simulation. The best-discovered loss function combinations **CSE-Autoloss-A** and **CSE-Autoloss-B** outperform default combinations (Cross-entropy/Focal loss for classification and L1 loss for regression) by 1.1\% and 0.8\% in terms of mAP for two-stage and one-stage detectors on COCO respectively.

The repository contains the demo training scripts for the best-searched loss combinations of our paper (ICLR2021) [Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search](https://openreview.net/pdf?id=5jzlpHvvRk). 

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Get Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.

## Searched Loss

#### Two-Stage Best-Discovered Loss

<img title="" src="img/CSE-Autoloss-A_cls.png" alt="avatar" width="594" data-align="center">

```python
CSE_Autoloss_A_cls='Neg(Dot(Mul(Y,Add(1,Sin(Z))),Log(Softmax(X))))'
```

<img title="" src="img/CSE-Autoloss-A_reg.png" alt="avatar" width="442" data-align="center">

```python
CSE_Autoloss_A_reg='Add(1,Neg(Add(Div(I,U),Neg(Div(Add(E,Neg(Add(I,2))),E)))))'
```

#### One-Stage Best-Discovered Loss

<img title="" src="img/CSE-Autoloss-B_cls.png" alt="avatar" data-align="center">

```python
CSE_Autoloss_B_cls='Neg(Add(Mul(Q,Mul(Add(1,Serf(Sig(NY))),Log(Sig(X)))),Mul(Add(Sgdf(X),Neg(Q)),Mul(Add(Add(1,Neg(Q)),Neg(Add(1,Neg(Sig(X))))),Log(Add(1,Neg(Sig(X))))))))'
```

<img title="" src="img/CSE-Autoloss-B_reg.png" alt="avatar" width="461" data-align="center">

```python
CSE_Autoloss_B_reg='Neg(Div(Add(Div(Neg(Add(Neg(E),Add(1,I))),Neg(Add(3,Add(2,U)))),Add(Div(E,E),Div(Neg(E),Neg(1)))),Neg(Add(Div(Neg(Add(U,Div(I,1))),Neg(3)),Neg(E)))))'
```

[1] *u*, *i*, *e*, *w* indicate union, intersection, enclose and intersection-over-union (IoU) between bounding box prediction and groundtruth. *x*, *y* are for class prediction and label.  
[2] *dot* is for dot product, *erf* is for scaled error function, *gd* is for scaled gudermannian function. Please see more details about "S"-shaped curve at [wiki](https://en.wikipedia.org/wiki/Sigmoid_function).

## Performance

Performance for COCO val are as follows.
Detector | Loss | Bbox mAP | Command | Checkpoint
--- |:---:|:---:|:---:|:---:
Faster R-CNN R50 | CSE-Autoloss-A | 38.5% | [Link](commands/faster_rcnn_r50_fpn_giou_1x_coco.sh) | [Link](https://drive.google.com/file/d/1WtKEJo5bKp4rKq6382CXs7dsd90sSWv1/view?usp=sharing)
Faster R-CNN R101 | CSE-Autoloss-A | 40.2% | [Link](commands/faster_rcnn_r101_fpn_giou_1x_coco.sh) | -
Cascade R-CNN R50 | CSE-Autoloss-A | 40.5% | [Link](commands/cascade_rcnn_r50_fpn_giou_1x_coco.sh) | -
Mask R-CNN R50 | CSE-Autoloss-A | 39.1% | [Link](commands/mask_rcnn_r50_fpn_giou_1x_coco.sh) | -
FCOS R50 | CSE-Autoloss-B | 39.6% | [Link](commands/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.sh) | [Link](https://drive.google.com/file/d/1c53qgqsIUnQ1EuOksGGfy7gdlh7PWR2S/view?usp=sharing)
ATSS R50 | CSE-Autoloss-B | 40.5% | [Link](commands/atss_r50_fpn_giou_1x_coco_w1.sh) | -

[1] We replace the centerness_target in FCOS and ATSS to the **IoU** between bbox_pred and bbox_target. Please see more details at [fcos_head.py](https://github.com/PerdonLiu/CSE-Autoloss/blob/b0a0ec56e3b531604683a8cc8e9df37a9cef3b0b/mmdet/models/dense_heads/fcos_head.py#L235-L239) and [atss_head.py](https://github.com/PerdonLiu/CSE-Autoloss/blob/b0a0ec56e3b531604683a8cc8e9df37a9cef3b0b/mmdet/models/dense_heads/atss_head.py#L196-L200).

[2] For the search loss combinations, loss_bbox weight for ATSS sets to 1 (instead of 2). Please see more details [here](configs/atss/atss_r50_fpn_giou_1x_coco_w1.py). 

## Quick start to train the model with searched/default loss combinations

```
# cls - classification, reg - regression

# Train with searched classification loss and searched regression loss
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --loss_cls $SEARCH_CLS_LOSS --loss_reg $SEARCH_REG_LOSS --launcher pytorch;

# Train with searched classification loss and default regression loss
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --loss_cls $SEARCH_CLS_LOSS --launcher pytorch;

# Train with default classification loss and searched regression loss
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --loss_reg $SEARCH_REG_LOSS --launcher pytorch;

# Train with default classification loss and default regression loss
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch;
```

## Acknowledgement

Thanks to MMDetection Team for their powerful deep learning detection framework. Thanks to Huawei Noah's Ark Lab AI Theory Group for their numerous V100 GPUs.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@inproceedings{liu2020loss,
  title={Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search},
  author={Liu, Peidong and Zhang, Gengwei and Wang, Bochao and Xu, Hang and Liang, Xiaodan and Jiang, Yong and Li, Zhenguo},
  booktitle={ICLR},
  year={2020}
}
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
