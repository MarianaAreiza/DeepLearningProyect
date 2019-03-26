# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Entrenamiento del sistema

#### 0 - Librerias utilizadas
Se utilizaron las siguientes librerías:
- **Pandas**. Es un paquete de Python que proporciona estructuras de datos similares a los dataframes de R.
- **NUMPY**. Es el encargado de añadir toda la capacidad matemática y vectorial a Python haciendo posible operar con cualquier dato numérico o array. Incorpora operaciones tan básicas como la suma o la multiplicación u otras mucho más complejas como la transformada de Fourier o el álgebra lineal. Además incorpora herramientas que nos permiten incorporar código fuente de otros lenguajes de programación como C/C++ o Fortran lo que incrementa notablemente su compatibilidad e implementación.
- **TensorFlow**. Es una librería de código abierto para cálculo numérico, usando como forma de programación grafos de flujo de datos. Los nodos en el grafo representan operaciones matemáticas, mientras que las conexiones o links del grafo representan los conjuntos de datos multidimensionales (tensores).
- **OS**. El módulo OS nos permite acceder a funcionalidades dependientes del Sistema Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular la estructura de directorios (para leer y escribir archivos).
- **TIME**. El módulo time de la biblioteca estándar de Python proporciona un conjunto de funciones para trabajar con fechas y/o horas.
```
import pandas as pd
import numpy as np
import tensorflow
import os
import time
```
#### 1 - Estructura del Sistema
```
from tensorflow.keras.layers import Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy,top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from sklearn.metrics import confusion_matrix
```
#### 2 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
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
```


