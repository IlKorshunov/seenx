from deepface.commons import weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


CONV_KW = dict(use_bias=False, kernel_initializer="glorot_normal")
BN_KW = dict(axis=3, epsilon=2e-5, momentum=0.9)
STAGES = [(64, 3), (128, 4), (256, 6), (512, 3)]


def conv_bn(x, filters, kernel, stride, name):
    if kernel > 1:
        x = ZeroPadding2D(padding=kernel // 2, name=f"{name}_pad")(x)
    x = Conv2D(filters, kernel, strides=stride, name=f"{name}_conv", **CONV_KW)(x)
    return BatchNormalization(name=f"{name}_bn", **BN_KW)(x)


def ir_block(x, filters, stride, conv_shortcut, name):
    shortcut = conv_bn(x, filters, 1, stride, f"{name}_0") if conv_shortcut else x
    x = BatchNormalization(name=f"{name}_1_bn", **BN_KW)(x)
    x = conv_bn(x, filters, 3, 1, f"{name}_1c")
    x = PReLU(shared_axes=[1, 2], name=f"{name}_1_prelu")(x)
    x = conv_bn(x, filters, 3, stride, f"{name}_2")
    return Add(name=f"{name}_add")([shortcut, x])


def stack(x, filters, blocks, stride, name):
    x = ir_block(x, filters, stride, conv_shortcut=True, name=f"{name}_block1")
    for i in range(2, blocks + 1):
        x = ir_block(x, filters, 1, conv_shortcut=False, name=f"{name}_block{i}")
    return x


def build_backbone(stages=STAGES, name="ResNet34"):
    inp = Input(shape=(112, 112, 3))
    x = conv_bn(inp, 64, 3, 1, "conv1")
    x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
    for idx, (filters, blocks) in enumerate(stages, start=2):
        x = stack(x, filters, blocks, stride=2, name=f"conv{idx}")
    return Model(inp, x, name=name)


def build_arcface(weight_file, stages=STAGES, embedding_dim=512, dropout=0.4):
    backbone = build_backbone(stages)
    x = BatchNormalization(momentum=0.9, epsilon=2e-5)(backbone.output)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(embedding_dim, kernel_initializer="glorot_normal")(x)
    x = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(x)
    model = Model(backbone.input, x, name=backbone.name)
    return weight_utils.load_model_weights(model=model, weight_file=weight_file)


class ArcFaceClient(FacialRecognition):
    def __init__(self, weight_file):
        self.model = build_arcface(weight_file)
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512
