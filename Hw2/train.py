import tensorflow as tf
import datetime, os
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

#%load_ext tensorboard

DATASET_PATH  ='PetImages'
IMAGE_SIZE =(128,128)
NUM_CLASSES = 2
BATCH_SIZE = 32
FREEZE_LAYERS = 2
NUM_EPOCHS = 5

train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',target_size=IMAGE_SIZE,interpolation='bicubic',class_mode='categorical',shuffle=True,batch_size=BATCH_SIZE)
valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',target_size=IMAGE_SIZE,interpolation='bicubic',class_mode='categorical',shuffle=False,batch_size=BATCH_SIZE)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

model_resnet = ResNet50(weights='imagenet', include_top=False)
input = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),name = 'image_input')
output_resnet = model_resnet(input)

x = Flatten(name='flatten')(output_resnet)
x = Dense(64, activation='relu', name='dense_1')(x)
x = Dense(64, activation='relu', name='dense_2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

my_model=Model(input,x)
my_model.summary()

sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
my_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer= sgd, metrics= ['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# 訓練模型
try:
  hist = my_model.fit(train_batches,steps_per_epoch = 20,validation_data = valid_batches,validation_steps = 4,epochs = NUM_EPOCHS,callbacks=[tensorboard_callback])
except Exception as e:
  print(e)

# 儲存訓練好的模型
my_model.save('original/train.h5')

fig = plt.figure()
plt.plot(hist.history['accuracy'],label='training')
plt.plot(hist.history['val_accuracy'],label='testing')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('original/accuracy.png')
fig = plt.figure()
plt.plot(hist.history['loss'],label='training loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('original/loss.png')

