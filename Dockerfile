# Usar una imagen base de Python
FROM python:3.10-slim

# Instalar timidity
RUN apt-get update && apt-get install -y timidity

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Establecer las variables de entorno
ENV FLASK_APP=run.py

# Copiar los archivos de la aplicación al contenedor
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8080 para Flask
EXPOSE 8080

# Definir el comando para ejecutar la aplicación Flask
CMD ["python", "run.py"]