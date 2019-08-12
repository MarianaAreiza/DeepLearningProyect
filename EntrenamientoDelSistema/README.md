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
import matplotlib.pyplot as plt
import tensorflow
import tensorflow as tf
import os
import itertools
import shutil
import matplotlib.pyplot as plt
import time
mport seaborn as sn
import scikitplot as skplt
```
#### 1 - Estructura del Sistema
```
from tensorflow.keras.applications.nasnet import NASNetLarge as PTModel, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,LearningRateScheduler
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from tensorflow.keras.layers import Activation, Dropout,Concatenate, Flatten, Dense, GlobalMaxPooling2D, GlobalAveragePooling2D,BatchNormalization, Input, Conv2D
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.metrics import roc_curve, auc
```
#### 2 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
train_path="img2/train"
valid_path="img2/val"


train_batch_size = 16
val_batch_size = 16
size = 224
steps=500
image_size = 224
num_class= 2

EPOCHS =5


datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

train_batches = datagen.flow_from_directory(train_path,target_size=(size,size),batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,target_size=(size,size),batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)
def get_model_classif_nasnet():
    inputs = Input((size, size, 3))
    base_model= DenseNet121(include_top=False, weights='imagenet',input_shape=(size, size, 3), classes=num_class)
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(num_class, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-9), loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model.summary()

    return model
model = get_model_classif_nasnet()

print(valid_batches.class_indices)

weightpath = "./model/weights-{epoch:03d}-{val_loss:.3f}.hdf5"
filepath = "model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,verbose=1,
                                mode='auto',epsilon=0.0001, cooldown=5, min_lr=0.0001)


tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), batch_size=train_batch_size, write_images=True)



callbacks_list = [checkpoint,reduce_lr,tensorboard]

history= model.fit_generator(train_batches, steps_per_epoch=steps,
                              class_weight=weightpath,
                    validation_data=valid_batches,
                    validation_steps=steps,
                    epochs=EPOCHS, verbose=1,
                   callbacks=callbacks_list)



print('SIGUIENTE ETAPA: VERIFICACION')


pred_Y = model.predict(valid_batches, batch_size = train_batch_size, verbose = True)
print('ya pase pred_Y')

model.metrics_names

print('comienza evaluacion: la ultima etapa')
val_loss, val_cat_acc = \
model.evaluate_generator(test_batches,
                        steps=11669)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)

print('evaluacion: La mejor etapa')

model.load_weights('model.h5')

val_loss, val_cat_acc = \
model.evaluate_generator(test_batches,
                        steps=11669)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)

print('EMPIEZA A IMPRIMIR PRECISION Y PERDIDA')

import matplotlib.pyplot as plt

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training cat acc')
plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
plt.title('Training and validation cat accuracy')
plt.legend()
plt.figure()

plt.show()

print('HE TERMINADO LA GRAFICA')

test_labels = test_batches.classes
test_batches.class_indices
predictions = model.predict_generator(test_batches, steps=11669, verbose=1)
predictions.shape

print('EMPEZANDO CON MATRIZ DE CONFUSION')

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
test_batches.class_indices
cm_plot_labels = ['masasYnodulos','sanas']

df_cm = pd.DataFrame(cm, range(num_class), range(num_class))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()

print('HE TERMINADO LA MATRIZ DE CONFUSION')

print('GENERADOR DE REPORTE DE CLASIFICACION')

y_pred = np.argmax(predictions, axis=1)
y_true =test_batches.classes

from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=cm_plot_labels)
print(report)
print('HE TERMINADO EL REPORTE DE CLASIFICACION')
print('ENTRENAMIENTO FINALIZADO')

#ROC CURVE
skplt.metrics.plot_roc(y_true,predictions,figsize=(20,14))
plt.show()
```


