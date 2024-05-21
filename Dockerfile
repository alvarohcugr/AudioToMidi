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

# Exponer el puerto 5000 para que Flask pueda escuchar
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0"]