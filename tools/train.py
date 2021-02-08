import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from tools.get_loss import build_binary_cls_loss, build_multi_cls_loss, build_reg_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--loss_cls', help='str to build cls loss function')
    parser.add_argument('--loss_reg', help='str to build reg loss function')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def rebuild_model_loss(model, loss_cls, loss_reg):
    if 'roi_head' in dir(model): # Two-Stage
        if loss_cls is not None and loss_cls != 'None' and loss_cls != 'none':
            if not isinstance(model.roi_head.bbox_head, torch.nn.modules.container.ModuleList): # Faster R-CNN
                cls_loss_weight = model.roi_head.bbox_head.loss_cls.loss_weight
                loss_cls_fun = build_multi_cls_loss(loss_cls, loss_weight=cls_loss_weight)
                model.roi_head.bbox_head.loss_cls = loss_cls_fun
            else:
                if len(model.roi_head.bbox_head) == 3: # Cascade R-CNN
                    cls_loss_weight = model.roi_head.bbox_head[0].loss_cls.loss_weight
                    loss_cls_fun = build_multi_cls_loss(loss_cls, loss_weight=cls_loss_weight)
                    model.roi_head.bbox_head[0].loss_cls = loss_cls_fun
                    model.roi_head.bbox_head[1].loss_cls = loss_cls_fun
                    model.roi_head.bbox_head[2].loss_cls = loss_cls_fun
                else:
                    raise NotImplementedError
        if loss_reg is not None and loss_reg != 'None' and loss_reg != 'none':
            if not isinstance(model.roi_head.bbox_head, torch.nn.modules.container.ModuleList): # Faster R-CNN
                reg_loss_weight = model.roi_head.bbox_head.loss_bbox.loss_weight
                loss_bbox_fun = build_reg_loss(loss_reg, loss_weight=reg_loss_weight)
                model.roi_head.bbox_head.loss_bbox = loss_bbox_fun
            else:
                if len(model.roi_head.bbox_head) == 3: # Cascade R-CNN
                    reg_loss_weight = model.roi_head.bbox_head[0].loss_bbox.loss_weight
                    loss_bbox_fun = build_reg_loss(loss_reg, loss_weight=reg_loss_weight)
                    model.roi_head.bbox_head[0].loss_bbox = loss_bbox_fun
                    model.roi_head.bbox_head[1].loss_bbox = loss_bbox_fun
                    model.roi_head.bbox_head[2].loss_bbox = loss_bbox_fun
                else:
                    raise NotImplementedError
    elif 'bbox_head' in dir(model): # One-Stage
        if loss_cls is not None and loss_cls != 'None' and loss_cls != 'none':
            cls_loss_weight = model.bbox_head.loss_cls.loss_weight
            model.bbox_head.loss_cls = build_binary_cls_loss(loss_cls, loss_weight=cls_loss_weight)
        if loss_reg is not None and loss_reg != 'None' and loss_reg != 'none':
            reg_loss_weight = model.bbox_head.loss_bbox.loss_weight
            model.bbox_head.loss_bbox = build_reg_loss(loss_reg, loss_weight=reg_loss_weight)
    else:
        raise NotImplementedError
    return model

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if args.loss_cls is None:
        logger.info('Train with default cls loss')
    else:
        logger.info(f'Train with auto loss cls: {args.loss_cls}')
    if args.loss_reg is None:
        logger.info('Train with default reg loss')
    else:
        logger.info(f'Train with auto loss reg: {args.loss_reg}')
    model = rebuild_model_loss(model, args.loss_cls, args.loss_reg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
