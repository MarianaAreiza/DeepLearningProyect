# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Mover Imagenes de una carpeta a otra por medio de documento CSV

#### 0 - Librerias utilizadas
Se utilizaron las siguientes librerías:
- **OS**. El módulo OS nos permite acceder a funcionalidades dependientes del Sistema Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular la estructura de directorios (para leer y escribir archivos).
- **Pandas**. Es un paquete de Python que proporciona estructuras de datos similares a los dataframes de R.
- **SYS**. El módulo sys es el encargado de proveer variables y funcionalidades, directamente relacionadas con el intérprete.
```
import os
import sys
import pandas as pd
```
#### 1 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
datos = data.loc[data.FindingLabels.str.startswith('Mass') & data.FindingLabels.str.endswith('Mass')].ImageIndex
#datos = data.loc[data.FindingLabels.str.startswith('Nodule', na=False)].ImageIndex
#datos = datos.loc[data.FindingLabels.str.contains('Mass')].ImageIndex

image=datos.tolist()


source='/home/longino/Documents/DATA/imagenOriginal/images/'
destination='/home/longino/Documents/DATA/imagenOriginal/SoloMass/'
for img in image:
    os.rename(source+img, destination+img)
    
print("Dataset folders successfully created by breed name and copied all images in corresponding folders")
```


