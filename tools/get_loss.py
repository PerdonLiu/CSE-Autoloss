import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss
from deap.gp import compile

from tools.search_space import get_pset


def binary_cls_loss(
        func,
        cls_pred,
        cls_label,
        pos_ious=None,
        weight=None,
        alpha=0.25,
        reduction="mean",
        avg_factor=None):
    bg_class_ind = cls_pred.shape[-1]
    # pos_inds for pos sample index
    pos_inds = ((cls_label >= 0) & (cls_label < bg_class_ind)).nonzero().reshape(-1)
    # label for binary target y, value is 1 for pos sample groundtruth class, others 0
    label = torch.zeros_like(cls_pred)
    label[pos_inds, cls_label[pos_inds]] = 1.0

    # value is IoU for pos sample groundtruth class, others 0
    iou = torch.zeros_like(cls_pred)
    if pos_ious is not None:
        pos_ious = pos_ious.detach()
        iou[pos_inds, cls_label[pos_inds]] = pos_ious
    loss = func(cls_pred, iou, label, 1 - label)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    if weight is not None and len(loss.size()) > 1:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def multi_cls_loss(
        func,
        cls_pred,
        cls_label,
        pos_ious=None,
        weight=None,
        class_weight=None,
        reduction='mean',
        avg_factor=None):
    bg_class_ind = cls_pred.shape[-1] - 1
    # 0 ~ self.num_classses-1 are FG, self.num_classses is BG
    num_class = cls_pred.shape[-1]
    # pos_inds for pos sample index
    pos_inds = (cls_label >= 0) & (cls_label < bg_class_ind)
    onehot_label = F.one_hot(cls_label, num_class)

    # value is IoU for pos sample groundtruth class, others 1
    iou = torch.ones_like(onehot_label).type(torch.cuda.FloatTensor)
    if pos_ious is not None:
        pos_ious = pos_ious.detach()
        iou[pos_inds.type(torch.bool), :] = pos_ious.unsqueeze(dim=1)

    loss = func(cls_pred, onehot_label, iou)
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def reg_loss(
        func,
        bbox_pred,
        bbox_targets,
        weight=None,
        linear=False,
        reduction='mean',
        avg_factor=None,
        eps=1e-7,
        **kwargs):
    # overlap
    lt = torch.max(bbox_pred[:, :2], bbox_targets[:, :2])
    rb = torch.min(bbox_pred[:, 2:], bbox_targets[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    # area / enclose
    ap = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    ag = (bbox_targets[:, 2] - bbox_targets[:, 0]) * (bbox_targets[:, 3] - bbox_targets[:, 1])
    enclose_x1y1 = torch.min(bbox_pred[:, :2], bbox_targets[:, :2])
    enclose_x2y2 = torch.max(bbox_pred[:, 2:], bbox_targets[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
    # union
    union = ap + ag - overlap + eps

    loss = func(union, overlap, enclose)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
  
class BinaryClsLoss(nn.Module):

    def __init__(self,
                 func,
                 use_sigmoid=True,
                 beta=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(BinaryClsLoss, self).__init__()
        assert use_sigmoid is True, 'only support use_sigmoid=True'
        self.func = func
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
  
    def forward(self,
                cls_pred,
                cls_label,
                pos_ious=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.loss_weight * binary_cls_loss(
                self.func,
                cls_pred,
                cls_label,
                pos_ious=pos_ious,
                weight=weight,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss

class MultiClsLoss(nn.Module):

    def __init__(self,
                 func,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(MultiClsLoss, self).__init__()
        self.func = func
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
  
    def forward(self,
                cls_pred,
                cls_label,
                pos_ious=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_pred.new_tensor(
                self.class_weight, device=cls_pred.device)
        else:
            class_weight = None
        loss = self.loss_weight * multi_cls_loss(
            self.func,
            cls_pred,
            cls_label,
            pos_ious=pos_ious,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

class RegLoss(nn.Module):

    def __init__(self,
                 func,
                 linear=False,
                 eps=1e-7,
                 reduction='mean',
                 loss_weight=1.0):
        super(RegLoss, self).__init__()
        self.eps = eps
        self.linear = linear
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.func = func
  
    def forward(self,
                bbox_pred,
                bbox_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if (weight is not None) and (not torch.any(weight > 0)) and (self.reduction != 'none'):
            return (bbox_pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        weight = weight.mean(-1)
        loss = self.loss_weight * reg_loss(
            self.func,
            bbox_pred,
            bbox_targets,
            weight=weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

def build_multi_cls_loss(loss_func, loss_weight=1.0):
    return MultiClsLoss(
                compile(loss_func, get_pset(mode='MULTI_CLS', arg_num=3)),
                reduction='mean',
                loss_weight=loss_weight)

def build_binary_cls_loss(loss_func, loss_weight=1.0):
    return BinaryClsLoss(
                compile(loss_func, get_pset(mode='BINARY_CLS', arg_num=4)),
                use_sigmoid=True,
                beta=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=loss_weight)

def build_reg_loss(loss_func, loss_weight=1.0):
    return RegLoss(
                compile(loss_func, get_pset(mode='REG', arg_num=3)),
                linear=False,
                eps=1e-7,
                reduction='mean',
                loss_weight=loss_weight)
