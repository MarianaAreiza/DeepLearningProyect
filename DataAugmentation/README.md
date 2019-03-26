# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Aaumentar nuestro Dataset por medio de algoritmos
#### 0 - Librerias utilizadas
Se utilizaron las siguientes librerías:
- **OS**. El módulo OS nos permite acceder a funcionalidades dependientes del Sistema Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular la estructura de directorios (para leer y escribir archivos).
- **SHUTIL**. Es especializado en operaciones de alto nivel para el manejo de documentos y directorios.
- **TensorFlow**. Es una librería de código abierto para cálculo numérico, usando como forma de programación grafos de flujo de datos. Los nodos en el grafo representan operaciones matemáticas, mientras que las conexiones o links del grafo representan los conjuntos de datos multidimensionales (tensores).
```
import os
import shutil
import tensorflow
```
#### 1 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
class_dir = ['masas','nodulos']

for item in class_dir:
    
    # archivo temporal para almacenar imagenes
    tmp_dir = 'tmp_dir'
    if not os.path.exists(tmp_dir):
    	os.makedirs(tmp_dir)

    class_dir = os.path.join(tmp_dir, 'class_dir')

    if not os.path.exists(class_dir):
    	os.makedirs(class_dir)
   

    # list all images in that directory
    img_list = os.listdir(item)

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for fname in img_list:
            # source path to image
            src = os.path.join(item, fname)
            # destination path to image
            dst = os.path.join(class_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


    # point to a dir containing the images and not to the images themselves
    path = tmp_dir
    save_path = item

    # Create a data generator
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                           save_to_dir=save_path,
                                           save_format='png',
                                                    target_size=(224,224),
                                                    batch_size=batch_size)



    # Generate the augmented images and add them to the training folders
    
    ###########
    
    num_total_img = 10000 # total number of images we want to have in each class
    
    ###########
    
    num_files = len(class_dir)
    num_batches = int(np.ceil((num_total_img-num_files)/batch_size))
    for i in range(0,num_batches):
    	imgs, labels = next(aug_datagen)

    shutil.rmtree('tmp_dir')

```


