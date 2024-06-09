import time
from flask import Flask, render_template, request, send_file
from predict import get_midi_from_wav
import torch
from model import UNet, CNN_Model
import os
import subprocess
from midi_processing import modify_midi_file
def midi_to_wav(midi_path, wav_path):
    # Usar timidity para convertir MIDI a WAV
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path])

app = Flask(__name__)

# Verificar si hay disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Función para cargar un modelo
def load_model(model_name):
    harmonics = [0.5, 1, 2, 3, 4, 5, 6, 7]
    if (model_name == 'checkpoint_unet'):
        model = UNet(n_classes=1, harmonics=harmonics).to(device)
        
    elif (model_name == 'checkpoint_deep'):
        model = CNN_Model(harmonics=harmonics).to(device)
    model.eval()
    model_path = f'{model_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    return model

# Cargar todos los modelos disponibles
models = {
    'checkpoint_unet': load_model('checkpoint_unet'),
    'checkpoint_deep': load_model('checkpoint_deep')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No se ha proporcionado un archivo de audio"

    audio_file = request.files['file']
    model_name = request.form['model']

    # Obtener el modelo seleccionado
    model = models.get(model_name)
    if not model:
        return "Modelo no válido seleccionado"
    
    # Generar un nombre de archivo único basado en la marca de tiempo actual
    timestamp = int(time.time())
    midi_file_name = f'output_{timestamp}.mid'
    wav_file_name = f'output_{timestamp}.wav'
    
    # Guardar el archivo en disco
    midi_file_path= f"outputs/{midi_file_name}"
    wav_file_path = f"outputs/{wav_file_name}"
    # Convertir piano roll a formato MIDI
    onsets=model_name=='checkpoint_deep'
    midi_file = get_midi_from_wav(audio_file, model, onsets)

    # Guardar el archivo MIDI en disco
    midi_file.save(midi_file_path)

    # Convertir MIDI a WAV
    midi_to_wav(midi_file_path, wav_file_path)

    # Devolver las URLs de los archivos MIDI y WAV al usuario
    midi_url = f"/get_midi/{midi_file_name}"
    wav_url = f"/get_wav/{wav_file_name}"

    return render_template('index.html', wav_url=wav_url, midi_url=midi_url)

@app.route('/modify_midi', methods=['POST'])
def modify_midi():
     # Obtener el camino del archivo MIDI a modificar desde la solicitud del formulario
    midi_url = request.form['midi_file_path']
    # Obtener el tempo e instrumento seleccionados por el usuario desde el formulario
    tempo = int(request.form['tempo'])
    instrument = int(request.form['instrument'])
    # Obtener las rutas locales a los archivos MIDI y WAV
    midi_filename = midi_url.split('/')[-1]
    wav_filename = midi_filename.replace('.mid', '.wav')
    midi_path = "outputs/"+midi_filename
    wav_path = "outputs/"+wav_filename
    # Modificar el archivo MIDI con el tempo e instrumento seleccionados
    modify_midi_file(midi_path, tempo, instrument)
    midi_to_wav(midi_path, wav_path)
    # Cambiar nombres de los archivos MIDI y WAV
    timestamp = int(time.time())
    os.rename(midi_path, f"outputs/output{timestamp}.mid")
    os.rename(wav_path, f"outputs/output{timestamp}.wav")
    # Devolver las URLs de los archivos MIDI y WAV al usuario
    wav_url = f"/get_wav/output{timestamp}.wav"
    midi_url = f"/get_midi/output{timestamp}.mid"

    return render_template('index.html', wav_url=wav_url, midi_url=midi_url)
@app.route('/get_midi/<filename>')
def get_midi(filename):
    # Enviar el archivo MIDI al usuario para su descarga
    response = send_file("outputs/"+filename, as_attachment=True)
    
    # Eliminar el archivo después de la descarga
    #os.remove(filename)
    
    return response

@app.route('/get_wav/<filename>')
def get_wav(filename):
    # Enviar el archivo WAV al usuario para su descarga
    response = send_file("outputs/"+filename, as_attachment=True, mimetype="audio/wav")
    
    # Eliminar el archivo después de la descarga
    #os.remove(filename)
    
    return response
if __name__ == '__main__':
    app.run(debug=True)
