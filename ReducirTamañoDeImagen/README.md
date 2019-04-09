# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Reducir el tamaño de las imágenes

#### 0 - Librerias utilizadas
Se utilizaron las siguientes librerías:

- **OS**. El módulo OS nos permite acceder a funcionalidades dependientes del Sistema Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular la estructura de directorios (para leer y escribir archivos).
- **SYS**. El módulo sys es el encargado de proveer variables y funcionalidades, directamente relacionadas con el intérprete.
- **Pandas**. Es un paquete de Python que proporciona estructuras de datos similares a los dataframes de R.
- **Pillow(PIL)**. Es una biblioteca gratuita para el lenguaje de programación Python que agrega soporte para abrir, manipular y guardar muchos formatos de archivos de imágenes diferentes.
```
import os
import sys
import pandas as pd
from PIL import Image
```
#### 1 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
data = pd.read_csv('/home/longino/Documents/DATA/Masas/mass.csv')

datos = data.ImageIndex

image=datos.tolist()

filename = '/home/longino/Documents/prueba/masasVAL/'


for img in image:
	file_parts = os.path.splitext(filename+img)
	outfile = file_parts[0] + file_parts[1]
	size = 224, 224
	try:
		img = Image.open(filename+img)
		img = img.resize(size, Image.ANTIALIAS)
		img.save(outfile, "PNG")
	except IOError as e:
		print("An exception ocurred '%s'" % e)

print("Dataset folders successfully created by breed name and copied all images in corresponding folders")
```


