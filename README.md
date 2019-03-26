# DeepLearningProyect

Este trabajo es un proyecto final para finalizar la tesis de grado en la Universidad Tecnológica de Panamá, Centro Regional de Chiriquí, Financiada por La Secretaría Nacional de Ciencia, Tecnología e Innovación de la República de Panamá (SENACYT) bajo el codigo del proyecto [APY-NI-2018-13](https://www.senacyt.gob.pa/wp-content/uploads/2018/04/ACTA-DE-RECEPCI%C3%93N-DE-PROPUESTAS-DE-NUEVOS-INVESTIGADORES-2018-RONDA-I.pdf).


## Primeros Pasos

#### 0 - Prerequisitos
- Instalar el sistema de gestión de paquetes y distribución de software [ANACONDA](https://www.anaconda.com/distribution/).
- Luego de instalar anaconda hay que crear un perfil con el siguiente código, en donde está escríto "deep" es el nombre que le quieras colocar a tu perfil:
```
conda create -n deep
```
- Luego de haber creado el perfíl hay que activarlo por medio de este código:
```
conda activate deep
```

#### 1 - Librerías
Luego del perfil hay que instalar las librerías necesarias para el funcionamiento del sistema las cuales serían las siguientes:
- **TensorFlow (CPU)**. ([TensorFlow](https://www.tensorflow.org/)) es una biblioteca de código abierto para aprendizaje automático a través de un rango de tareas, en este caso es el CPU el cuál solo utilizara el procesador del computador para realizar el trabajo.
```
conda install -c conda-forge tensorflow
```
- **TensorFlow (GPU)**. Para la utilización de la tarjeta gráfica se necesita instalar primero el driver de ([NVIDIA](https://www.nvidia.es/Download/index.aspx?lang=es)). **Solo es permitido para las tarjetas gŕaficas de marca NVIDIA ya que cuentan con los núcleos CUDA que son los que permiten la utilización de estas tarjetas para el desarrollo de proyectos de ciencia de datos.**

Si se encuentra en un sistema operativo basado en Linux lo más recomendable es utilizar la distribución ([Antergos](https://antergos.com/)) por que por defecto instala los driver de NVIDIA, por lo tanto el siguiente paso sería instalar el driver CUDA por medio del siguiente código:
```
yaourt -S nvidia nvidia-utils cuda
```

- **Matplotlib**. Es una biblioteca de gráficos 2D en Python que produce cifras de calidad de publicación en una variedad de formatos de copia impresa y entornos interactivos en todas las plataformas, para la instalación de esta sería el siguiente código:
```
conda install -c conda-forge matplotlib 
```

- **Pandas**. Es una librería de python destinada al análisis de datos, que proporciona unas estructuras de datos flexibles y que permiten trabajar con ellos de forma muy eficiente.
```
conda install -c conda-forge pandas
```

- **openCV**. Es una biblioteca libre de visión artificial originalmente desarrollada por Intel.
```
conda install -c conda-forge opencv
```

#### 2 - PrimerPrograma

Luego de instalar todo lo antes mencionado hay que instalar el editor de texto, en ese caso, utilizaremos SublimeText por su flexibilidad y facilidad de uso con diferentes entornos.
```
https://www.sublimetext.com/3
```

luego hay que configurarlo para que funcione correctamente con ANACONDA, para eso, presionaremos la tecla:
```
Control + SHIFT + P
```
Luego se les abrirá un menu plegable y deberán escribir "install", presionar enter, les deber[ia aparecer un mensaje de finalizado, luego abrir nuevamente el menú plegable y escribir "install", les deberá aparecer otras opciones, en este caso presional el que dice "Package Cotrol: Install Package", les abrirá un menú similar al anterior y luego escriben "CONDA", deberán presionar enter donde solo aparezca "CONDA", les abrirá una pestaña con un texto, eso significa que ya está instalado, luego debén presionar en el menú de arriba "Tools/Build System/Conda".

Luego presionan el atajo del menu plegable nuevamente y escriben "CONDA" y presionar enter donde aparezca "CONDA: Activate Enviorment" y les aparecera dos opciones, uno que se llama "root" y otro que se llama "deep" (o como le hayan llamado ustedes a su perfil).

deben hacer este ultimo paso cada vez que abran un archivo.py
