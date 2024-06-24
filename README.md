# AudioToMidi
AudioToMidi es una aplicación de Transcripción Automática de la Música que permite introducir ficheros de audio y obtener a cambio un archivo .MIDI con el resultado de la transcripción, utilizando modelos de aprendizaje automático. Nota: La aplicación está pensada la transcripción de archivos de audio de piano. El uso de archivos de audio con otro tipo de instrumentos puede dar lugar a una menor calidad en la transcripción.

Esta aplicación ha sido creada por Álvaro Hernández Coronel para el Trabajo de Fin de Grado del grado de Ingeniería Informática en la Escuela Técnica Superior de Ingeniería Informática y Telecomunicaciones (UGR). El TFG ha sido tutorizado por Miguel García Silvente y Eugenio Aguirre Molina.

## Opción 1. Ejecución local
### Paso 1: Asegúrate de tener Python y pip instalados:
Puedes verificar la instalación con:
```bash 
python --version 
pip --version
```
### Opcional: Entorno virtual
Si quieres un entorno aislado para las dependencias de python, puedes crear un entorno virtual con Virtualenv.
Instala y activa el entorno:
```bash
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
```
Ahora tus dependencias se instalarán en el entorno virtual. Puedes salir del entorno virtual con el comando `deactivate`
### Paso 2: Instalar dependencias    
Ejecuta el script `install.sh`

### Paso 3: Ejecutar  
Ejecuta el script `run.sh`
Esto iniciará la aplicación Flask y estará accesible en `http://localhost:8080`


## Opción 2. Ejecución con Docker

### Paso 1: Construir la imagen Docker

Asegúrate de tener Docker instalado en tu máquina. Desde la terminal, navega hasta el directorio donde se encuentra el fichero Dockerfile y ejecuta el siguiente comando para construir la imagen Docker:
`docker build -t nombre_de_tu_imagen .`


Sustituye `nombre_de_tu_imagen` por el nombre que deseas darle a tu imagen Docker.

### Paso 2: Ejecutar el contenedor Docker

Una vez que la imagen se haya construido correctamente, puedes ejecutar el contenedor Docker con el siguiente comando:

`docker run -d -p 8080:8080 nombre_de_tu_imagen`


Esto iniciará tu aplicación Flask en un contenedor Docker y estará accesible en `http://localhost:8080`.

### Detener el contenedor Docker

Para detener el contenedor Docker en ejecución, puedes usar los siguientes comandos:

```bash
docker ps  -a
docker stop CONTAINER_ID
```

Sustituye CONTAINER_ID con el ID del contenedor que deseas detener, que puedes encontrar en la salida del comando `docker ps -a`.

# Entrenamiento de modelos 
## Google Colab
El fichero TFG_Colab.ipynb contiene el código de entrenamiento y evaluación de los modelos en el dataset MAPS.

## Kaggle
Para entrenar y evaluar los dos modelos finales de la aplicación en el dataset MAESTRO, se han usado dos Notebooks, uno para entrenamiento y otro para evaluación en el conjunto de prueba. En el historial de versiones se pueden ver los resultados y el código de las ejecuciones.
### Entrenamiento
Notebook TFG_MAESTRO. Versión 13 --> Entrenamiento modelo BP2, 30 épocas: 
`https://www.kaggle.com/code/alvarohernandezc/tfg-maestro?scriptVersionId=176879493`

Notebook TFG_MAESTRO. Versión 19 --> Entrenamiento modelo UNetH, 13 épocas: `https://www.kaggle.com/code/alvarohernandezc/tfg-maestro?scriptVersionId=181816880`

Notebook TFG_MAESTRO. Versión 22 --> Continuación entrenamiento del modelo UNetH, 11 épocas: `https://www.kaggle.com/code/alvarohernandezc/tfg-maestro?scriptVersionId=183657756`



### Evaluación. 
Notebook TFG_Test_MAESTRO. Versión 11 --> Evaluación modelo BP2: `https://www.kaggle.com/code/alvarohernandezc/tfg-test-maestro?scriptVersionId=182810550`

Notebook TFG_Test_MAESTRO. Versión 9 --> Evaluación modelo UNetH: `https://www.kaggle.com/code/alvarohernandezc/tfg-test-maestro?scriptVersionId=182675366`