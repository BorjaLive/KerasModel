from tensorflow.keras import datasets, layers, models, preprocessing, activations, Input, initializers, optimizers
import tensorflow as tf
from model import zesNet
from dataset import TrafficLightDataset

BATCH_SIZE = 16
MAX_EPOCHS = 800
INIT_LR = 0.001
WEIGHT_DECAY = 0.00005
LR_DROP_MILESTONES = [400,600]

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

train_file_dir = './Annotations/training_file.csv'
valid_file_dir = './Annotations/validation_file.csv'
train_img_dir = './Annotations/PTL_Dataset_876x657'
valid_img_dir = './Annotations/PTL_Dataset_768x576'
save_path = './new_weights_v1'

train_dataset = TrafficLightDataset(csv_file = train_file_dir, img_dir = train_img_dir, batch_size=BATCH_SIZE)
valid_dataset = TrafficLightDataset(csv_file = valid_file_dir, img_dir = valid_img_dir, batch_size=BATCH_SIZE)

model = zesNet()

adam = optimizers.Adam() #learning_rate=0.001, clipnorm=1
model.compile(loss='binary_crossentropy', optimizer=adam, metrics="accuracy")
model.fit(train_dataset, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
#model.summary()

model.save('model.h5')
