import sys
import math
import random
from deap import gp
import torch
import torch.nn.functional as F
eps = 1e-20
pi = torch.tensor([math.pi])


def get_pset(mode='MULTI_CLS', arg_num=3):
    if mode.upper() == 'MULTI_CLS':
        return get_multi_cls_pset(arg_num)
    elif mode.upper() == 'BINARY_CLS':
        return get_binary_cls_pset(arg_num)
    elif mode.upper() == 'REG':
        return get_reg_pset(arg_num)
    else:
        raise NotImplementedError

def get_binary_cls_pset(arg_num=4):
    assert arg_num == 4, 'arg_num must be 4'
    pset = gp.PrimitiveSet('CLS', 4)

    # Element-Wise Op
    pset.addPrimitive(Neg, 1)
    pset.addPrimitive(Exp, 1)
    pset.addPrimitive(Log, 1)
    pset.addPrimitive(Abs, 1)
    pset.addPrimitive(Sqrt, 1)
    pset.addPrimitive(Softmax, 1)
    pset.addPrimitive(Softplus, 1)
    pset.addPrimitive(Sig, 1)
    pset.addPrimitive(Sgdf, 1)
    pset.addPrimitive(Salf, 1)
    pset.addPrimitive(Serf, 1)
    pset.addPrimitive(Tanh, 1)
    pset.addPrimitive(Sin, 1)
    pset.addPrimitive(Cos, 1)
    pset.addPrimitive(Relu, 1)
    pset.addPrimitive(Add, 2)
    pset.addPrimitive(Sub, 2)
    pset.addPrimitive(Mul, 2)
    pset.addPrimitive(Div, 2)
    # Constant
    pset.addEphemeralConstant('rand_constant_cls', lambda: random.choice([1, 2, 3]))

    # name for input values
    # X for cls_pred, Q for iou scores, PY for binary target y, NY for 1-y
    pset.renameArguments(ARG0='X', ARG1='Q', ARG2='PY', ARG3='NY')
    return pset

def get_multi_cls_pset(arg_num=3):
    assert arg_num in [2, 3], 'arg_num must be 2 or 3'
    pset = gp.PrimitiveSet('CLS', arg_num)

    # Element-Wise Op
    pset.addPrimitive(Neg, 1)
    pset.addPrimitive(Exp, 1)
    pset.addPrimitive(Log, 1)
    pset.addPrimitive(Abs, 1)
    pset.addPrimitive(Sqrt, 1)
    pset.addPrimitive(Softmax, 1)
    pset.addPrimitive(Softplus, 1)
    pset.addPrimitive(Sig, 1)
    pset.addPrimitive(Sgdf, 1)
    pset.addPrimitive(Salf, 1)
    pset.addPrimitive(Serf, 1)
    pset.addPrimitive(Tanh, 1)
    pset.addPrimitive(Sin, 1)
    pset.addPrimitive(Cos, 1)
    pset.addPrimitive(Relu, 1)
    pset.addPrimitive(Add, 2)
    pset.addPrimitive(Sub, 2)
    pset.addPrimitive(Mul, 2)
    pset.addPrimitive(Div, 2)
    # Reduce Op
    pset.addPrimitive(Dot, 2)
    # Constant
    pset.addEphemeralConstant('rand_constant_cls', lambda: random.choice([1, 2, 3]))

    # name for input values
    # X for cls_pred, Y for cls ont-hot label, Z for iou scores
    if arg_num == 2:
        pset.renameArguments(ARG0='X', ARG1='Y')
    elif arg_num == 3:
        pset.renameArguments(ARG0='X', ARG1='Y', ARG2='Z')
    return pset

def get_reg_pset(arg_num=3):
    assert arg_num == 3, 'arg_num must be 3'
    pset = gp.PrimitiveSet('REG', 3)

    # Element-Wise Op
    pset.addPrimitive(Add, 2)
    pset.addPrimitive(Sub, 2)
    pset.addPrimitive(Mul, 2)
    pset.addPrimitive(Neg, 1)
    pset.addPrimitive(Log, 1)
    pset.addPrimitive(Exp, 1)
    pset.addPrimitive(Div, 2)
    pset.addPrimitive(Abs, 1)
    pset.addPrimitive(Sqrt, 1)
    pset.addPrimitive(Softplus, 1)
    pset.addPrimitive(Sig, 1)
    pset.addPrimitive(Tanh, 1)
    pset.addPrimitive(Relu, 1)
    pset.addPrimitive(Sin, 1)
    pset.addPrimitive(Cos, 1)
    # Constant
    pset.addEphemeralConstant('rand_constant_reg', lambda: random.choice([1, 2, 3]))

    # name for input values
    # U for union, I for intersect, E for enclose
    pset.renameArguments(ARG0='U', ARG1='I', ARG2='E')
    return pset

def toCuda(x):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor([x])
    return x.cuda()

# Reduce Op
def Dot(x1, x2):
    x1 = toCuda(x1)
    x2 = toCuda(x2)
    mul = torch.mul(x1, x2)
    if len(mul.shape) == 1:
        ret = torch.sum(mul, dim=0)
    else:
        ret = torch.sum(mul, dim=1)
    return ret

# Element-Wise Op
def Add(x1, x2):
    x1 = toCuda(x1)
    x2 = toCuda(x2)
    ret = x1 + x2
    return ret

def Sub(x1, x2):
    x1 = toCuda(x1)
    x2 = toCuda(x2)
    ret = x1 - x2
    return ret

def Sqrt(x):
    x = toCuda(x)
    ret = torch.sqrt(x.float())
    return ret

def Mul(x1, x2):
    x1 = toCuda(x1)
    x2 = toCuda(x2)
    ret = torch.mul(x1, x2)
    return ret

def Log(x):
    x = toCuda(x)
    ret = torch.log(x + eps)
    return ret

def Neg(x):
    x = toCuda(x)
    ret = -x
    return ret

def Div(x1, x2):
    x1 = toCuda(x1)
    x2 = toCuda(x2)
    ret = x1 / (x2 + eps)
    return ret

def Exp(x):
    x = toCuda(x)
    # avoid overflow
    x = torch.clamp(x, max=88.)
    ret = torch.exp(x.float())
    return ret

def Abs(x):
    x = toCuda(x)
    ret = torch.abs(x)
    return ret

def Sig(x):
    x = toCuda(x)
    ret = torch.sigmoid(x.float())
    return ret

def Serf(x):
    x = toCuda(x)
    x = x.float()
    x = (torch.sqrt(pi.cuda()) / 2) * x / 2
    x = torch.erf(x)
    ret = (x + 1) / 2
    return ret

def Salf(x):
    x = toCuda(x)
    x = x.float() / 2
    x = x / torch.sqrt(1 + torch.pow(x, 2))
    ret = (x + 1) / 2
    return ret

def Sgdf(x):
    x = toCuda(x)
    x = x.float() * pi.cuda() / 4
    # Gudermannian function
    x = 2 * torch.atan(torch.tanh(x / 2))
    x = x * 2 / pi.cuda()
    ret = (x + 1) / 2
    return ret

def Tanh(x):
    x = toCuda(x)
    ret = torch.tanh(x.float())
    return ret

def Relu(x):
    x = toCuda(x)
    ret = torch.relu(x.float())
    return ret

def Sin(x):
    x = x.cuda()
    ret = torch.sin(x.float())
    return ret

def Cos(x):
    x = x.cuda()
    ret = torch.cos(x.float())
    return ret

def Softmax(x):
    x = x.cuda()
    ret = torch.softmax(x.float(), dim=1)
    return ret

def Softplus(x):
    x = toCuda(x)
    ret = F.softplus(x.float(), beta=1)
    return ret