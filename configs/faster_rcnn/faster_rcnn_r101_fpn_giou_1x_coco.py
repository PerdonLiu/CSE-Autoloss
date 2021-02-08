_base_ = './faster_rcnn_r50_fpn_giou_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101),
            roi_head=dict(
                bbox_head=dict(
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                )
            )
        )
