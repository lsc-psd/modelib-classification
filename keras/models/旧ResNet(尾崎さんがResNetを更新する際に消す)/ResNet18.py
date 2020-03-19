from .ResNet_base import make_base_model, make_bottleneck_model

def build(input_shape, num_class):
    return make_base_model(input_shape, num_class, [2, 2, 2, 2])