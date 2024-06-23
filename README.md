# Nombre del Proyecto

Breve descripción o introducción de tu proyecto aquí.

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