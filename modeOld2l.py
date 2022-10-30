#import numpy as np

#import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, activations, Input

# La imagen de entrada debe ser múltiplo de 64
input = Input(shape=(3, 768, 576), name="image")

#Primera capa
xx1 = layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first")(input)
x1 = layers.Conv2D(strides = (2, 2), filters = 32, kernel_size = 3, use_bias = False, data_format = "channels_first", name = "Conv2d-1")(xx1)
x2 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-2")(x1)
x3 = layers.ReLU(6, name = "ReLU6-3")(x2)
x4 = layers.MaxPooling2D(pool_size = (2, 2), data_format = "channels_first", name="MaxPool2d-4")(x3)

#Bottleneck expansion_ratio = 1, output_channels = 16, n = 1, stride = 1
xx5 = layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first")(x4)
x5 = layers.Conv2D(groups = 32, filters = 32, kernel_size = 3, use_bias = False, data_format = "channels_first", name="Conv2d-5")(xx5)
x6 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-6")(x5)
x7 = layers.ReLU(6, name="ReLU6-7")(x6)
x8 = layers.Conv2D(filters = 16, kernel_size = 1, use_bias = False, data_format = "channels_first", name="Conv2d-8")(x7)
x9 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-9")(x8)

#Bottleneck expansion_ratio = 6, output_channels = 24, n = 2, stride = 2
x11 = layers.Conv2D(filters = 96, kernel_size = 1, use_bias = False, data_format = "channels_first", name = "Conv2d-11")(x9)
x12 = layers.BatchNormalization(axis = 1, name = "BatchNorm2d-12")(x11)
x13 = layers.ReLU(6, name = "ReLU6-13")(x12)
xx14 = layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first")(x13)
x14 = layers.Conv2D(strides = (2, 2), groups = 96, filters = 96, kernel_size = 3, use_bias = False, data_format = "channels_first", name="Conv2d-14")(xx14)
x15 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-15")(x14)
x16 = layers.ReLU(6, name="ReLU6-16")(x15)
x17 = layers.Conv2D(filters = 24, kernel_size = 1, use_bias = False, data_format = "channels_first", name="Conv2d-17")(x16)
x18 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-18")(x17)

x20 = layers.Conv2D(filters = 144, kernel_size = 1, use_bias = False, data_format = "channels_first", name = "Conv2d-20")(x18)
x21 = layers.BatchNormalization(axis = 1, name = "BatchNorm2d-21")(x20)
x22 = layers.ReLU(6, name = "ReLU6-22")(x21)
xx23 = layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first")(x22)
x23 = layers.Conv2D(groups = 144, filters = 144, kernel_size = 3, use_bias = False, data_format = "channels_first", name="Conv2d-23")(xx23)
x24 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-24")(x23)
x25 = layers.ReLU(6, name="ReLU6-25")(x24)
x26 = layers.Conv2D(filters = 24, kernel_size = 1, use_bias = False, data_format = "channels_first", name="Conv2d-26")(x25)
x27 = layers.BatchNormalization(axis = 1, name="BatchNorm2d-27")(x26)

x28 = layers.Add(name="InvertedResidual-28")([x18, x27])

model = models.Model(input, x28, name = "LytNet")
model.summary()




"""
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

print(tf.test.gpu_device_name())

modelName = "mnist.h5"
model = models.load_model(modelName)

model.summary()

def get_model():
    model = models.Sequential(name = "mnist-CNN-tf")
    model.add(layers.Conv2D(8, 3, padding = 'same', activation='relu', input_shape=(28, 28, 1), name = "layer1"))
    model.add(layers.AveragePooling2D((2, 2), name = "layer2"))
    model.add(layers.Conv2D(13, 3, padding = 'same', activation='relu', name = "layer3"))
    model.add(layers.Flatten(name = "layer4"))
    model.add(layers.Dense(14*14*13, name = "layer5"))
    model.add(layers.Dense(10, name = "layer6"))
    return model
    
model = get_model()
model.summary()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

print(datetime.now())
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
print(datetime.now())

predictions = model.predict(x_test[:10])
predictions.argmax(axis=1), y_test[:10]

model.save('mnist.h5')
"""

"""
def convert_to_tflite(model, filename):
    # Convert the tensorflow model into a tflite file.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

mnist_tflite_filename = "mnist.tflite"
convert_to_tflite(model, mnist_tflite_filename)
"""

"""
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

predictions = model.predict(x_test[:10])
result = predictions.argmax(axis=1)
print(result, y_test[:10])
show_img(x_test[6])
"""

