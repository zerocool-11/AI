#!/usr/bin/env python
# coding: utf-8

from keras.layers import Convolution2D
from keras.models import Sequential
import numpy as np 
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D

model = Sequential()

model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64, 64, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=250,
        epochs=40,
        validation_data=test_set,
        validation_steps=800)


model.save('dog_cat.h5')

test_image = image.load_img('cnn_dataset/single_prediction/cat_!.jpeg', 
               target_size=(64,64))


test_image = image.img_to_array(test_image)


test_image = np.expand_dims(test_image, axis=0)

result = m.predict(test_image)


if result[0][0] == 1.0:
    print('dog')
else:
    print('cat')


r = training_set.class_indices


