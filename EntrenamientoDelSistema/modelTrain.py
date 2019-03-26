import pandas as pd
import numpy as np
import tensorflow
import os
import time
from tensorflow.keras.layers import Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy,top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from sklearn.metrics import confusion_matrix



train_path="img/train"
valid_path="img/val"

train_batch_size = 10
val_batch_size = 10
size = 224
steps= 600

num_class= 2

EPOCHS = 2

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

train_batches = datagen.flow_from_directory(train_path,target_size=(size,size),batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,target_size=(size,size),batch_size=val_batch_size)

base_model= DenseNet121(include_top=False, weights='imagenet',input_shape=(size, size, 3), classes=num_class)


base_model.summary()

len(base_model.layers)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(lr=1e-4, decay=1e-9), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

model.summary()


reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), batch_size=train_batch_size, write_images=True)

weightpath = "./model/weights-{epoch:03d}-{top_3_accuracy:.3f}.hdf5"
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')
                              
                              
callbacks_list = [checkpoint, reduce_lr,tensorboard]

history = model.fit_generator(train_batches, steps_per_epoch=steps, 
                              class_weight=weightpath,
                    validation_data=valid_batches,
                    validation_steps=steps,
                    epochs=30, verbose=1,
                   callbacks=callbacks_list)

