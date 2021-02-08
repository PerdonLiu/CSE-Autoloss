# CSE-Autoloss

Designing proper loss functions for vision tasks has been a long-standing research direction to advance the capability of existing models. For object detection, the well-established classification and regression loss functions have been carefully designed by considering diverse learning challenges (e.g. class imbalance, hard negative samples, and scale variances). Inspired by the recent progress in network architecture search, it is interesting to explore the possibility of discovering new loss function formulations via directly searching the primitive operation combinations. So that the learned losses not only fit for diverse object detection challenges to alleviate huge human efforts, but also have better alignment with evaluation metric and good mathematical convergence property. Beyond the previous auto-loss works on face recognition and image classification, our work makes the first attempt to discover new loss functions for the challenging object detection from primitive operation levels and finds the searched losses are insightful. We propose an effective convergence-simulation driven evolutionary search algorithm, called CSE-Autoloss, for speeding up the search progress by regularizing the mathematical rationality of loss candidates via two progressive convergence simulation modules: convergence property verification and model optimization simulation. The best-discovered loss function combinations **CSE-Autoloss-A** and **CSE-Autoloss-B** outperform default combinations (Cross-entropy/Focal loss for classification and L1 loss for regression) by 1.1\% and 0.8\% in terms of mAP for two-stage and one-stage detectors on COCO respectively.

The repository contains the demo training scripts for the best-searched loss combinations of our paper (ICLR2021) [Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search](https://openreview.net/forum?id=5jzlpHvvRk). 

## Installation
Please refer to [get_started.md](docs/get_started.md) for installation.

## Get Started
Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.

## Searched Loss 

#### Two-Stage Best-Discovered Loss

<p href="https://www.codecogs.com/eqnedit.php?latex=\text&space;{&space;CSE-Autoloss-A}_{\text&space;{cls&space;}}(x,&space;y,&space;w)=-(1&plus;\sin&space;(w))&space;y&space;\log&space;(\operatorname{softmax}(x))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\text&space;{&space;CSE-Autoloss-}A_{\text&space;{cls&space;}}(x,&space;y,&space;w)=-(1&plus;\sin&space;(w))&space;y&space;\log&space;(\operatorname{softmax}(x))" title="\text { CSE-Autoloss-A}_{\text {cls }}(x, y, w)=-(1+\sin (w)) y \log (\operatorname{softmax}(x))" /></p>

```python
CSE_Autoloss_A_cls='Neg(Dot(Mul(Y,Add(1,Sin(Z))),Log(Softmax(X))))'
```

<p href="https://www.codecogs.com/eqnedit.php?latex=\text&space;{&space;CSE-Autoloss-A}_{\text&space;{reg&space;}}(i,&space;u,&space;e)=\left(1-\frac{i}{u}\right)&plus;\left(1-\frac{i&plus;2}{e}\right)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\text&space;{&space;CSE-Autoloss-A}_{\text&space;{reg&space;}}(i,&space;u,&space;e)=\left(1-\frac{i}{u}\right)&plus;\left(1-\frac{i&plus;2}{e}\right)" title="\text { CSE-Autoloss-A}_{\text {reg }}(i, u, e)=\left(1-\frac{i}{u}\right)+\left(1-\frac{i+2}{e}\right)" /></p>

```python
CSE_Autoloss_A_reg='Add(1,Neg(Add(Div(I,U),Neg(Div(Add(E,Neg(Add(I,2))),E)))))'
```

#### One-Stage Best-Discovered Loss

<p href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\tiny&space;\text&space;{&space;CSE-Autoloss-B}_{\mathrm{cls}}(x,&space;y,&space;w)=-[w&space;y(1&plus;\operatorname{erf}(\sigma(1-y)))&space;\log&space;\sigma(x)&plus;(\operatorname{gd}(x)-w&space;y)(\sigma(x)-w&space;y)&space;\log&space;(1-\sigma(x))]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\tiny&space;\text&space;{&space;CSE-Autoloss-B}_{\mathrm{cls}}(x,&space;y,&space;w)=-[w&space;y(1&plus;\operatorname{erf}(\sigma(1-y)))&space;\log&space;\sigma(x)&plus;(\operatorname{gd}(x)-w&space;y)(\sigma(x)-w&space;y)&space;\log&space;(1-\sigma(x))]" title="\tiny \text { CSE-Autoloss-B}_{\mathrm{cls}}(x, y, w)=-[w y(1+\operatorname{erf}(\sigma(1-y))) \log \sigma(x)+(\operatorname{gd}(x)-w y)(\sigma(x)-w y) \log (1-\sigma(x))]" /></p>

```python
CSE_Autoloss_B_cls='Neg(Add(Mul(Q,Mul(Add(1,Serf(Sig(NY))),Log(Sig(X)))),Mul(Add(Sgdf(X),Neg(Q)),Mul(Add(Add(1,Neg(Q)),Neg(Add(1,Neg(Sig(X))))),Log(Add(1,Neg(Sig(X))))))))'
```

<p href="https://www.codecogs.com/eqnedit.php?latex=\text&space;{&space;CSE-Autoloss-B}_{\text&space;{reg&space;}}(i,&space;u,&space;e)=\frac{3&space;e&space;u&plus;12&space;e&plus;3&space;i&plus;3&space;u&plus;18}{-3&space;e&space;u&plus;i&space;u&plus;u^{2}-15&space;e&plus;5&space;i&plus;5&space;u}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\text&space;{&space;CSE-Autoloss-B}_{\text&space;{reg&space;}}(i,&space;u,&space;e)=\frac{3&space;e&space;u&plus;12&space;e&plus;3&space;i&plus;3&space;u&plus;18}{-3&space;e&space;u&plus;i&space;u&plus;u^{2}-15&space;e&plus;5&space;i&plus;5&space;u}" title="\text { CSE-Autoloss-B}_{\text {reg }}(i, u, e)=\frac{3 e u+12 e+3 i+3 u+18}{-3 e u+i u+u^{2}-15 e+5 i+5 u}" /></p>

```python
CSE_Autoloss_B_reg='Neg(Div(Add(Div(Neg(Add(Neg(E),Add(1,I))),Neg(Add(3,Add(2,U)))),Add(Div(E,E),Div(Neg(E),Neg(1)))),Neg(Add(Div(Neg(Add(U,Div(I,1))),Neg(3)),Neg(E)))))'
```

[1] *u*, *i*, *e*, *w* indicate union, intersection, enclose and intersection-over-union (IoU) between bounding box prediction and groundtruth. *x*, *y* are for class prediction and label.  
[2] *erf* is for scaled error function, *gd* is for scaled gudermannian function. Please see more details about "S"-shaped curve at [wiki](https://en.wikipedia.org/wiki/Sigmoid_function).

## Performance
Performance for COCO val are as follows.
Detector | Loss | Bbox mAP | Command
--- |:---:|:---:|:---:
Faster R-CNN R50 | CSE-Autoloss-A | 38.5% | [Link](commands/faster_rcnn_r50_fpn_giou_1x_coco.sh)
Faster R-CNN R101 | CSE-Autoloss-A | 40.2% | [Link](commands/faster_rcnn_r101_fpn_giou_1x_coco.sh)
Cascade R-CNN R50 | CSE-Autoloss-A | 40.5% | [Link](commands/cascade_rcnn_r50_fpn_giou_1x_coco.sh)
Mask R-CNN R50 | CSE-Autoloss-A | 39.1% | [Link](commands/mask_rcnn_r50_fpn_giou_1x_coco.sh)
FCOS R50 | CSE-Autoloss-B | 39.6% | [Link](commands/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.sh)
ATSS R50 | CSE-Autoloss-B | 40.5% | [Link](commands/atss_r50_fpn_giou_1x_coco_w1.sh)

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
Thanks to MMDetection Team for maintaining such a powerful open source project!

## Citation
If you use this toolbox or benchmark in your research, please cite this project.


```
@inproceedings{
  liu2021loss,
  title={Loss Function Discovery for Object Detection via Convergence-Simulation Driven Search},
  author={Peidong Liu and Gengwei Zhang and Bochao Wang and Hang Xu and Xiaodan Liang and Yong Jiang and Zhenguo Li},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=5jzlpHvvRk}
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