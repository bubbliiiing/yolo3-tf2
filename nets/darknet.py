from functools import wraps

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU,
                          ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose

#------------------------------------------------------#
#   单次卷积
#   DarknetConv2D
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
#   残差结构
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后对num_blocks进行循环，循环内部是残差结构。
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, weight_decay=5e-4):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2), weight_decay=weight_decay)(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1), weight_decay=weight_decay)(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3), weight_decay=weight_decay)(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x, weight_decay=5e-4):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3), weight_decay=weight_decay)(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3

