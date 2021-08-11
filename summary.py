#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body

if __name__ == "__main__":
    input_shape = [416, 416, 3]
    num_anchar  = 3
    num_classes = 80

    model = yolo_body(input_shape, num_anchar, num_classes)
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
