from datasetloader import load_dataset
from tensorflow.keras import datasets, layers, models, preprocessing, activations, Input, initializers
import tensorflow as tf
from model import zesNet
import numpy as np

BATCH_SIZE = 16
MAX_EPOCHS = 800

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

train_file_dir = './Annotations/training_file.csv'
valid_file_dir = './Annotations/validation_file.csv'
train_img_dir = './Annotations/PTL_Dataset_876x657'
valid_img_dir = './Annotations/PTL_Dataset_768x576'
save_path = './new_weights_v1'

x, y, z, w = load_dataset(train_file_dir, train_img_dir, total = 1000)

model = zesNet()

x = np.array(x)
y = np.array(y)
z = np.array(z)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics="accuracy")
model.fit(x, [y, z], epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

model.save('model.h5')