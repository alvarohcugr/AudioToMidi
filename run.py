from flask import Flask, render_template, request, send_file
from predict import get_midi_from_wav
import torch
from model import CNN_Model
import os
import subprocess

def midi_to_wav(midi_path, wav_path):
    # Usar timidity para convertir MIDI a WAV
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path])

app = Flask(__name__)

input_size=252
output_size=88
# Verificar si hay disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Crear la red y moverla a la GPU si está disponible
harmonics=[0.5, 1, 2, 3, 4, 5, 6, 7]
model = CNN_Model(harmonics).to(device)
model.eval()
# Ruta con los pesos del modelo
model_path = 'bp_all_7_no_logits.pth'
# Carga los pesos al modelo
model.load_state_dict(torch.load(model_path, map_location=device))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No se ha proporcionado un archivo de audio"

    audio_file = request.files['file']
    
    # Convertir piano roll a formato MIDI
    midi_file = get_midi_from_wav(audio_file, model)

    # Guardar el archivo MIDI en disco
    midi_file_path = 'output.mid'
    midi_file.save(midi_file_path)

    # Convertir MIDI a WAV
    wav_file_path = 'output.wav'
    midi_to_wav(midi_file_path, wav_file_path)

    # Devolver las URLs de los archivos MIDI y WAV al usuario
    midi_url = f"/get_midi/{midi_file_path}"
    wav_url = f"/get_wav/{wav_file_path}"
    
    return render_template('index.html', wav_url=wav_url, midi_url=midi_url)

@app.route('/get_midi/<filename>')
def get_midi(filename):
    return send_file(filename, as_attachment=True)

@app.route('/get_wav/<filename>')
def get_wav(filename):
    return send_file(filename, as_attachment=True, mimetype="audio/wav")

if __name__ == '__main__':
    app.run(debug=True)
