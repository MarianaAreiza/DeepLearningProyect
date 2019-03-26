# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Etiquetar Imagen

#### 0 - Librerias utilizadas
Se utilizaron las siguientes librerías:
- **Pandas**. Es un paquete de Python que proporciona estructuras de datos similares a los dataframes de R.
- **OpenCV(CV2)**. Es una librería de visión por computador de código abierto, ha sido diseñado para ser eficiente en cuanto a gasto de recursos computacionales y con un enfoque hacia las aplicaciones de tiempo real.
- **OS**. El módulo OS nos permite acceder a funcionalidades dependientes del Sistema Operativo. Sobre todo, aquellas que nos refieren información sobre el entorno del mismo y nos permiten manipular la estructura de directorios (para leer y escribir archivos).
- **TIME**. El módulo time de la biblioteca estándar de Python proporciona un conjunto de funciones para trabajar con fechas y/o horas.
```
import pandas as pd
import cv2
import os
import time
```
#### 1 - Estructura del Sistema
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
final=[]
drawing = False # true if mouse is pressed
ix,iy = -1,-1
def draw_rec(event,x,y,flags,param):
    global ix,iy,drawing,final

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
        w=x-ix
        h=y-iy
        raw_data = {'ImageIndex': file,'x': ix,'y': iy,'w': w,'h': h}
        final.append(raw_data)
        print(raw_data)
path = '/home/longino/Documents/Nodulos/Nodule1/'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            df=path+file
            img=cv2.imread(df)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',draw_rec)
            while(1):
                cv2.imshow('image',img)
                k = cv2.waitKey(1) & 0xFF
                if k == 13:
                    break
            cv2.destroyAllWindows()
df = pd.DataFrame(final, columns = ['ImageIndex', 'x', 'y', 'w', 'h'])
df.to_csv('example.csv')
print("Finalizado")
```


