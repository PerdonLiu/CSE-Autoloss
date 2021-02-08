cd /path/to/CSE-Autoloss
CSE_Autoloss_A_cls='Neg(Dot(Mul(Y,Add(1,Sin(Z))),Log(Softmax(X))))'
CSE_Autoloss_A_reg='Add(1,Neg(Add(Div(I,U),Neg(Div(Add(E,Neg(Add(I,2))),E)))))'
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29143 ./tools/train.py ./configs/cascade_rcnn/cascade_rcnn_r50_fpn_giou_1x_coco.py --loss_cls $CSE_Autoloss_A_cls --loss_reg $CSE_Autoloss_A_reg --launcher pytorch;