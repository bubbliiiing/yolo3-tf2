import tensorflow as tf
from tensorflow.keras import backend as K


#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def get_grid_anchors(feats, anchors):
    num_anchors = len(anchors)
    #------------------------------------------#
    #   grid_shape指的是特征层的高和宽
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #---------------------------------------------------------------#
    #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(tf.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])
    #------------------------------------------#
    #   获得各个特征点的坐标信息。
    #------------------------------------------#
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    #------------------------------------------#
    #   将各个特征点和先验框进行堆叠
    #------------------------------------------#
    grid_anchors = K.concatenate([K.cast(grid_x, K.dtype(feats)), K.cast(grid_y, K.dtype(feats)), K.cast(anchors_tensor, K.dtype(feats))])
    grid_anchors = K.reshape(grid_anchors, [-1, 4])
    #------------------------------------------#
    #   获得特征层的高宽
    #------------------------------------------#
    grid_w  = K.ones_like(grid_x) * grid_shape[1]
    grid_h  = K.ones_like(grid_y) * grid_shape[0]
    grid_wh = K.concatenate([K.cast(grid_w, K.dtype(feats)), K.cast(grid_h, K.dtype(feats))])
    grid_wh = K.reshape(grid_wh, [-1, 2])
    return grid_anchors, grid_wh

def decode_anchors(reshape_outputs, grid_anchors, grid_wh, input_shape):
    #------------------------------------------#
    #   对先验框进行解码，并进行归一化
    #------------------------------------------#
    box_xy          = (K.sigmoid(reshape_outputs[..., :2]) + grid_anchors[:, :2])/grid_wh
    box_wh          = K.exp(reshape_outputs[..., 2:4]) * grid_anchors[:, 2:] / K.cast(input_shape[::-1], K.dtype(reshape_outputs))
    #------------------------------------------#
    #   获得预测框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(reshape_outputs[..., 4:5])
    box_class_probs = K.sigmoid(reshape_outputs[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def DecodeBox(outputs,
            anchors,
            num_classes,
            input_shape,
            #-----------------------------------------------------------#
            #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
            #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
            #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
    
    image_shape = K.reshape(outputs[-1],[-1])

    #-----------------------------------------------------------#
    #   对特征层进行循环
    #   grid_anchors    代表每一个特征层所对应的先验框
    #   grid_whs        代表每一个特征层所对应的宽高
    #   reshape_outputs 代表YOLO网络的输出
    #-----------------------------------------------------------#
    grid_anchors    = []
    grid_whs        = []
    reshape_outputs = []
    for l in range(len(outputs) - 1):
        grid_anchor, grid_wh = get_grid_anchors(outputs[l], anchors[anchor_mask[l]])
        grid_anchors.append(grid_anchor)
        grid_whs.append(grid_wh)

        reshape_outputs.append(K.reshape(outputs[l], [-1, num_classes + 5]))
        
    grid_anchors    = K.concatenate(grid_anchors, axis = 0)
    reshape_outputs = K.concatenate(reshape_outputs, axis = 0)
    grid_whs        = K.concatenate(grid_whs, axis = 0)

    box_xy, box_wh, box_confidence, box_class_probs = decode_anchors(reshape_outputs, grid_anchors, grid_whs, input_shape)
    
    #------------------------------------------------------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条，因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对其进行修改，去除灰条的部分。 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #   如果没有使用letterbox_image也需要将归一化后的box_xy, box_wh调整成相对于原图大小的
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是：框的位置，得分与种类
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out
