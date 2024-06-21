import time
from flask import Flask, after_this_request, render_template, request, send_file
from predict import get_midi_from_wav
import torch
from model import UNet, CNN_Model
import os
import subprocess
from midi_processing import modify_midi_file
import tempfile
from memory_profiler import memory_usage
import psutil
import time

app = Flask(__name__)

# Verificar si hay disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def profile_memory(func):
    def wrapper(*args, **kwargs):
        # Obtener el uso de memoria inicial
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # Convertir a MB
        
        # Medir el tiempo de ejecución
        start_time = time.time()
        
        # Ejecutar la función y medir el uso de memoria
        mem_usage = memory_usage((func, args, kwargs), interval=0.1)
        result = func(*args, **kwargs)
        
        # Obtener el uso de memoria final
        mem_after = process.memory_info().rss / 1024 / 1024  # Convertir a MB
        
        end_time = time.time()

        print(f"Memoria inicial: {mem_before:.2f} MB")
        print(f"Memoria final: {mem_after:.2f} MB")
        print(f"Memoria máxima usada: {max(mem_usage):.2f} MB")
        print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        
        return result
    return wrapper

def midi_to_wav(midi_path, wav_path):
    # Usar timidity para convertir MIDI a WAV
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path])
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

def remove_outputs_folder():
    # Eliminar todos los archivos de salida
    for filename in os.listdir("outputs"):
        file_path = os.path.join("outputs", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

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

    remove_outputs_folder()

    audio_file = request.files['file']
    model_name = request.form['model']

    print("Transcribiendo el archivo:", audio_file.filename)
    # Obtener el modelo seleccionado
    model = models.get(model_name)
    if not model:
        return "Modelo no válido seleccionado"

    timestamp = int(time.time())
    midi_file_name = f'output_{timestamp}.mid'
    wav_file_name = f'output_{timestamp}.wav'
    midi_file_path = f"outputs/{midi_file_name}"
    wav_file_path = f"outputs/{wav_file_name}"

    # Guardar el archivo de audio en una ubicación temporal
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(temp_fd, 'wb') as temp_file:
        audio_file.save(temp_file)

    # Convertir piano roll a formato MIDI
    onsets = model_name == 'checkpoint_deep'
    
    # Decorar la función para medir memoria
    @profile_memory
    def transcribe_audio(temp_path, model, onsets):
        midi_file = get_midi_from_wav(temp_path, model, onsets)
        return midi_file

    midi_file = transcribe_audio(temp_path, model, onsets)

    # Guardar el archivo MIDI en disco
    midi_file.save(midi_file_path)

    # Convertir MIDI a WAV
    midi_to_wav(midi_file_path, wav_file_path)

    # Eliminar el archivo temporal
    os.remove(temp_path)

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
    filepath = f"outputs/{filename}"
    return send_file(filepath, as_attachment=True)

@app.route('/get_wav/<filename>')
def get_wav(filename):
    filepath = f"outputs/{filename}"
    return send_file(filepath, as_attachment=True, mimetype="audio/wav")
if __name__ == '__main__':
    app.run(debug=True)
