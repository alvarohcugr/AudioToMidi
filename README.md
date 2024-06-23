# AudioToMidi
AudioToMidi es una aplicación de Transcripción Automática de la Música que permite introducir ficheros de audio y obtener a cambio un archivo .MIDI con el resultado de la transcripción. Nota: La aplicación está pensada la transcripción de archivos de audio de piano. El uso de archivos de audio con otro tipo de instrumentos puede dar lugar a una menor calidad en la transcripción.

Esta aplicación ha sido creada por Álvaro Hernández Coronel para el Trabajo de Fin de Grado del grado de Ingeniería Informática en la Escuela Técnica Superior de Ingeniería Informática y Telecomunicaciones (UGR). El TFG ha sido tutorizado por Miguel García Silvente y Eugenio Aguirre Molina.

## Ejecutar
### Paso 1: Asegúrate de tener Python y pip instalados:
Puedes verificar la instalación con:
```bash 
python --version 
pip --version
```
### Paso 2: Instalar dependencias    
Ejecuta el script `install.sh`

### Paso 3: Ejecutar aplicación 
Ejecuta el script `run.sh`
Esto iniciará la aplicación Flask y estará accesible en `http://localhost:8080`


## Ejecutar con Docker

### Paso 1: Construir la imagen Docker

Asegúrate de tener Docker instalado en tu máquina. Desde la terminal, navega hasta el directorio donde se encuentra tu Dockerfile y ejecuta el siguiente comando para construir la imagen Docker:
`docker build -t nombre_de_tu_imagen` .


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