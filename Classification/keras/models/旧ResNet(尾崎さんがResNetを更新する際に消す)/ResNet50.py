from .ResNet_base import make_base_model, make_bottleneck_model

def build(input_shape, num_class):
    return make_bottleneck_model(input_shape, num_class, [3, 4, 6, 3])
