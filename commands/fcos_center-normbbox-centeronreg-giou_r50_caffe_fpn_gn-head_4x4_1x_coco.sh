cd /path/to/CSE-Autoloss
CSE_Autoloss_B_cls='Neg(Add(Mul(Q,Mul(Add(1,Serf(Sig(NY))),Log(Sig(X)))),Mul(Add(Sgdf(X),Neg(Q)),Mul(Add(Add(1,Neg(Q)),Neg(Add(1,Neg(Sig(X))))),Log(Add(1,Neg(Sig(X))))))))'
CSE_Autoloss_B_reg='Neg(Div(Add(Div(Neg(Add(Neg(E),Add(1,I))),Neg(Add(3,Add(2,U)))),Add(Div(E,E),Div(Neg(E),Neg(1)))),Neg(Add(Div(Neg(Add(U,Div(I,1))),Neg(3)),Neg(E)))))'
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29145 ./tools/train.py ./configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.py --loss_cls $CSE_Autoloss_B_cls --loss_reg $CSE_Autoloss_B_reg --launcher pytorch;