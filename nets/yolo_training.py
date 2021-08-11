import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils_bbox import decode_anchors, get_grid_anchors


#---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
#---------------------------------------------------#
def box_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   计算左上角的坐标和右下角的坐标
    #---------------------------------------------------#
    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   计算左上角和右下角的坐标
    #---------------------------------------------------#
    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #---------------------------------------------------#
    #   计算重合面积
    #---------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

#---------------------------------------------------#
#   loss值计算
#---------------------------------------------------#
def yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, ignore_thresh=.5, print_loss=False):
    num_layers = len(anchors_mask)
    #---------------------------------------------------------------------------------------------------#
    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    #   y_true是一个列表，包含三个特征层，shape分别为:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #---------------------------------------------------------------------------------------------------#
    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   得到input_shpae为416,416 
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    batch_size  = K.cast(K.shape(yolo_outputs[0])[0], K.dtype(y_true[0]))
    

    #-----------------------------------------------------------#
    #   取出每一张图片
    #   m的值就是batch_size
    #-----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    #-----------------------------------------------------------#
    #   对特征层进行循环
    #   grid_anchors    代表每一个特征层所对应的先验框
    #   grid_whs        代表每一个特征层所对应的宽高
    #   reshape_outputs 代表YOLO网络的输出
    #   reshape_y_true  代表真实框的情况
    #-----------------------------------------------------------#
    grid_anchors    = []
    grid_whs        = []
    reshape_outputs = []
    reshape_y_true  = []
    for l in range(len(anchors_mask)):
        grid_anchor, grid_wh = get_grid_anchors(yolo_outputs[l], anchors[anchors_mask[l]])
        grid_anchors.append(grid_anchor)
        grid_whs.append(grid_wh)
        
        reshape_outputs.append(K.reshape(yolo_outputs[l], [batch_size, -1, num_classes + 5]))
        reshape_y_true.append(K.reshape(y_true[l], [batch_size, -1, num_classes + 5]))

    grid_anchors    = K.concatenate(grid_anchors, axis = 0)
    grid_whs        = K.concatenate(grid_whs, axis = 0)
    reshape_outputs = K.concatenate(reshape_outputs, axis = 1)
    reshape_y_true  = K.concatenate(reshape_y_true, axis = 1)

    #----------------------------------------------------------#
    #   取出该特征层中存在目标的点的位置。(m,num_anchor,1)
    #-----------------------------------------------------------#
    object_mask      = reshape_y_true[..., 4:5]
    #-----------------------------------------------------------#
    #   取出其对应的种类(m,num_anchor,80)
    #-----------------------------------------------------------#
    true_class_probs = reshape_y_true[..., 5:]
    #-----------------------------------------------------------#
    #   将yolo_outputs的特征层输出进行处理，对先验框进行解码！
    #   pred_xy     (m,num_anchor,2) 解码后的中心坐标
    #   pred_wh     (m,num_anchor,2) 解码后的宽高坐标
    #-----------------------------------------------------------#
    box_xy, box_wh, _, _ = decode_anchors(reshape_outputs, grid_anchors, grid_whs, input_shape)
    #-----------------------------------------------------------#
    #   pred_box是解码后的预测的box的位置 (m,num_anchor,4)
    #-----------------------------------------------------------#
    pred_box = K.concatenate([box_xy, box_wh])
    #-----------------------------------------------------------#
    #   找到负样本群组，第一步是创建一个数组，[]
    #-----------------------------------------------------------#
    ignore_mask         = tf.TensorArray(K.dtype(y_true[0]), size = 1, dynamic_size = True)
    object_mask_bool    = K.cast(object_mask, 'bool')
    #-----------------------------------------------------------#
    #   对每一张图片计算ignore_mask
    #-----------------------------------------------------------#
    def loop_body(b, ignore_mask):
        #-----------------------------------------------------------#
        #   取出n个真实框：n,4
        #-----------------------------------------------------------#
        true_box = tf.boolean_mask(reshape_y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
        #-----------------------------------------------------------#
        #   计算预测框与真实框的iou
        #   pred_box    (num_anchor,4) 预测框的坐标
        #   true_box    (n,4)          真实框的坐标
        #   iou         (num_anchor,n) 预测框和真实框的iou
        #-----------------------------------------------------------#
        iou = box_iou(pred_box[b], true_box)

        #-----------------------------------------------------------#
        #   best_iou    (num_anchor,) 每个特征点与真实框的最大重合程度
        #-----------------------------------------------------------#
        best_iou = K.max(iou, axis=-1)

        #-----------------------------------------------------------#
        #   将与真实框重合度小于0.5以下的预测框对应的先验框作为负样本
        #   当预测框和真实框重合度较大时，不宜作为负样本。
        #   因为这些框已经预测的比较准确了
        #-----------------------------------------------------------#
        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
        return b + 1, ignore_mask

    #-----------------------------------------------------------#
    #   在这个地方进行一个循环、循环是对每一张图片进行的
    #-----------------------------------------------------------#
    _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

    #-----------------------------------------------------------#
    #   ignore_mask用于提取出作为负样本的特征点
    #   (m,num_anchor)
    #-----------------------------------------------------------#
    ignore_mask = ignore_mask.stack()
    #   (m,num_anchor,1)
    ignore_mask = K.expand_dims(ignore_mask, -1)
    #-----------------------------------------------------------#
    #   将真实框进行编码，使其格式与预测的相同，后面用于计算loss
    #-----------------------------------------------------------#
    raw_true_xy = reshape_y_true[..., :2] * grid_whs - grid_anchors[..., :2]
    raw_true_wh = K.log(reshape_y_true[..., 2:4] / grid_anchors[..., 2:] * input_shape[::-1])
    #-----------------------------------------------------------#
    #   object_mask如果真实存在目标则保存其wh值
    #   switch接口，就是一个if/else条件判断语句
    #-----------------------------------------------------------#
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
    #-----------------------------------------------------------#
    #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
    #   表示真实框的宽高，二者均在0-1之间
    #   真实框越大，比重越小，小框的比重更大。
    #-----------------------------------------------------------#
    box_loss_scale = 2 - reshape_y_true[...,2:3] * reshape_y_true[...,3:4]
    #-----------------------------------------------------------#
    #   利用binary_crossentropy计算中心点偏移情况，效果更好
    #-----------------------------------------------------------#
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, reshape_outputs[...,0:2], from_logits=True)
    #-----------------------------------------------------------#
    #   wh_loss用于计算宽高损失
    #-----------------------------------------------------------#
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - reshape_outputs[...,2:4])
    
    #------------------------------------------------------------------------------#
    #   如果该位置本来有框，那么计算1与置信度的交叉熵
    #   如果该位置本来没有框，那么计算0与置信度的交叉熵
    #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
    #   该操作的目的是：
    #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
    #   不适合当作负样本，所以忽略掉。
    #------------------------------------------------------------------------------#
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, reshape_outputs[...,4:5], from_logits=True) + \
                (1 - object_mask) * K.binary_crossentropy(object_mask, reshape_outputs[...,4:5], from_logits=True) * ignore_mask
    
    class_loss      = object_mask * K.binary_crossentropy(true_class_probs, reshape_outputs[...,5:], from_logits=True)

    #-----------------------------------------------------------#
    #   将所有损失求和
    #-----------------------------------------------------------#
    xy_loss         = K.sum(xy_loss)
    wh_loss         = K.sum(wh_loss)
    confidence_loss = K.sum(confidence_loss)
    class_loss      = K.sum(class_loss)
    #-----------------------------------------------------------#
    #   计算正样本数量
    #-----------------------------------------------------------#
    num_pos = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
    loss    = xy_loss + wh_loss + confidence_loss + class_loss
    if print_loss:
        loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
    loss = loss / num_pos
    return loss
