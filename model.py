#import numpy as np

#import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, activations, Input, initializers

xavierInitializer = initializers.GlorotNormal()
normalInitializer = initializers.RandomNormal(mean=0.0, stddev=0.01)

def first_layer(x, inch):
    x = layers.ZeroPadding2D(padding = (1, 1))(x)
    x = layers.Conv2D(strides = (2, 2), filters = inch, kernel_size = 3, use_bias = False, kernel_initializer=xavierInitializer)(x)
    x = layers.BatchNormalization(axis = 3, )(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPooling2D(pool_size = (2, 2))(x)
    return x

def bottleneck_layer(x, inch, outch, stride, expansion_ratio):
    hidden_dimms = round(inch * expansion_ratio)

    if expansion_ratio == 1:
        x = layers.ZeroPadding2D(padding = (1, 1))(x)
        x = layers.Conv2D(groups = hidden_dimms, filters = inch, kernel_size = 3, use_bias = False, kernel_initializer=xavierInitializer)(x)
        x = layers.BatchNormalization(axis = 3)(x)
        x = layers.ReLU(6)(x)
        x = layers.Conv2D(filters = outch, kernel_size = 1, use_bias = False, kernel_initializer=xavierInitializer)(x)
        x = layers.BatchNormalization(axis = 3)(x)
    else:
        x = layers.Conv2D(filters = hidden_dimms, kernel_size = 1, use_bias = False, kernel_initializer=xavierInitializer)(x)
        x = layers.BatchNormalization(axis = 3)(x)
        x = layers.ReLU(6)(x)
        x = layers.ZeroPadding2D(padding = (1, 1))(x)
        x = layers.Conv2D(strides =  (stride, stride), groups = hidden_dimms, filters = hidden_dimms, kernel_size = 3, use_bias = False, kernel_initializer=xavierInitializer)(x)
        x = layers.BatchNormalization(axis = 3)(x)
        x = layers.ReLU(6)(x)
        x = layers.Conv2D(filters = outch, kernel_size = 1, use_bias = False, kernel_initializer=xavierInitializer)(x)
        x = layers.BatchNormalization(axis = 3)(x)

    return x

def last_layer(x, outch):
    x = layers.Conv2D(filters = outch, kernel_size = 1, use_bias = False, kernel_initializer=xavierInitializer)(x)
    x = layers.BatchNormalization(axis = 3)(x)
    x = layers.ReLU(6)(x)
    return x

def zesNet(size_x = 768, size_y = 576):

    assert size_x % 64 == 0
    assert size_y % 64 == 0

    input_channels = 32
    last_channels = 1280

    inverted_residual_settings = [
        # t, c, n, s
        [1, 16, 1, 1], #  0     expansion_ratio = 1, output_channels = 16, n = 1, stride = 1
        [6, 24, 2, 2], #  1
        [6, 32, 1, 2], #  2
        [6, 64, 3, 2], #  3
        [6, 96, 1, 1], #  4
        [6, 160, 2, 2], # 5
        [6, 320, 1, 1], # 6
    ]

    # La imagen de entrada debe ser múltiplo de 64
    input = Input(shape=(size_x, size_y, 3), name="image")

    #Primera capa
    x = first_layer(input, inch = input_channels)

    #Bottlenecks
    for t, c, n, s in inverted_residual_settings:
        for i in range(n):
            if i == 0:
                x = bottleneck_layer(x, inch = input_channels, outch = c, stride = s, expansion_ratio = t)
            else:
                x = bottleneck_layer(x, inch = input_channels, outch = c, stride = 1, expansion_ratio = t)
            if i > 0:
                x = layers.Add()([shortcut, x])
            shortcut = x
            input_channels = c

    #Última capa
    x = last_layer(x, outch = last_channels)
    x = layers.Dropout(0.1)(x)

    x = layers.GlobalAveragePooling2D()(x)

    #Clasificacion
    light = layers.Dense(160, activation=None, kernel_initializer=normalInitializer)(x)
    light = layers.BatchNormalization()(light)
    light = layers.ReLU(6)(light)
    light = layers.Dense(5, activation=None, kernel_initializer=normalInitializer)(light)
    light = layers.Softmax(name="light")(light)

    #Regresion para la direccion
    direction = layers.Dense(80, activation=None, kernel_initializer=normalInitializer)(x)
    direction = layers.BatchNormalization()(direction)
    direction = layers.ReLU(6)(direction)
    direction = layers.Dense(4, activation=None, kernel_initializer=normalInitializer, name="direction")(direction)

    model = models.Model(inputs = input, outputs = [light, direction], name = "ZesNet")

    return model




