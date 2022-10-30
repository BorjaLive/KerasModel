#import numpy as np

#import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, activations

# La imagen de entrada debe ser múltiplo de 64

model = models.Sequential(name = "LytNet")

#Primera capa
model.add(layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first"))
model.add(layers.Conv2D(strides = (2, 2), filters = 32, kernel_size = 3, use_bias = False, data_format = "channels_first"))
model.add(layers.BatchNormalization(axis = 1))
model.add(layers.ReLU(6))
model.add(layers.MaxPooling2D(pool_size = (2, 2), data_format = "channels_first"))

#Bottleneck expansion_ratio = 1, output_channels = 16, n = 1, stride = 1
model.add(layers.ZeroPadding2D(padding = (1, 1), data_format = "channels_first"))
model.add(layers.Conv2D(groups = 32, filters = 32, kernel_size = 3, use_bias = False, data_format = "channels_first"))


model.build(input_shape = (None, 3, 768, 576))
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

