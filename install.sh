#!/bin/bash

# Install dependencies from requirements.txt
echo "Instalando dependencias de requirements.txt..."
pip install -r requirements.txt

echo "Instalando Timidity..."
# Instalación de Timidity
sudo apt-get update
sudo apt-get install -y timidity
echo "Dependencias instaladas."
