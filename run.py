import time
from flask import Flask, render_template, request, send_file
from predict import get_midi_from_wav
import torch
from model import UNet, CNN_Model
import os
from midi_processing import modify_midi_file, midi_to_wav
import tempfile
from memory_profiler import memory_usage
import psutil

app = Flask(__name__)

# Verificar si hay disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def profile_memory(func):
    """
    Decorador para medir el uso de memoria y el tiempo de ejecución de una función.

    Args:
        func (callable): Función a decorar.

    Returns:
        callable: Función decorada.
    """
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

# Función para cargar un modelo
def load_model(model_name):
    """
    Carga un modelo preentrenado de acuerdo al nombre especificado.

    Args:
        model_name (str): Nombre del modelo a cargar ('checkpoint_unet' o 'checkpoint_deep').

    Returns:
        torch.nn.Module: Modelo cargado en el dispositivo (CPU o GPU).
    """
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
    """
    Elimina todos los archivos en el directorio 'outputs'.

    Returns:
        None
    """
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
    """
    Renderiza la plantilla index.html en la ruta raíz.

    Returns:
        str: Contenido HTML de la plantilla renderizada.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Procesa la solicitud POST para transcribir un archivo de audio a MIDI y WAV.

    Returns:
        str: Contenido HTML de la plantilla index.html con URLs para descargar los archivos WAV y MIDI.
    """
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

    # Guardar el archivo de audio en una ubicación temporal (requerido para el perfilado de memoria)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(temp_fd, 'wb') as temp_file:
        audio_file.save(temp_file)

    # Convertir piano roll a formato MIDI
    onsets = model_name == 'checkpoint_deep'
    
    # Decorar la función para medir memoria
    @profile_memory
    def transcribe_audio(temp_path, model, onsets):
        """
        Función interna para transcribir un archivo de audio a MIDI.

        Args:
            temp_path (str): Ruta al archivo de audio temporal en formato WAV.
            model (torch.nn.Module): Modelo de red neuronal utilizado para la transcripción.
            onsets (bool): Indica si el modelo predice onsets o no.

        Returns:
            mido.MidiFile: Archivo MIDI generado.
        """
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
    """
    Procesa la solicitud POST para modificar un archivo MIDI con tempo e instrumento seleccionados.

    Returns:
        str: Contenido HTML de la plantilla index.html con URLs para descargar los archivos WAV y MIDI modificados.
    """

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
    output_midi =modify_midi_file(midi_path, tempo, instrument)

    # Cambiar nombres de los archivos MIDI y WAV
    timestamp = int(time.time())
    midi_path=f"outputs/output{timestamp}.mid"
    wav_path=f"outputs/output{timestamp}.wav"
    # Guardar el archivo MIDI modificado
    output_midi.save(midi_path)
    midi_to_wav(midi_path, wav_path)

    # Devolver las URLs de los archivos MIDI y WAV al usuario
    wav_url = f"/get_wav/output{timestamp}.wav"
    midi_url = f"/get_midi/output{timestamp}.mid"

    return render_template('index.html', wav_url=wav_url, midi_url=midi_url)
@app.route('/get_midi/<filename>')
def get_midi(filename):
    """
    Descarga un archivo MIDI desde el directorio 'outputs'.

    Args:
        filename (str): Nombre del archivo MIDI a descargar.

    Returns:
        flask.Response: Archivo MIDI como respuesta para descarga.
    """
    filepath = f"outputs/{filename}"
    return send_file(filepath, as_attachment=True)

@app.route('/get_wav/<filename>')
def get_wav(filename):
    """
    Descarga un archivo WAV desde el directorio 'outputs'.

    Args:
        filename (str): Nombre del archivo WAV a descargar.

    Returns:
        flask.Response: Archivo WAV como respuesta para descarga.
    """
    filepath = f"outputs/{filename}"
    return send_file(filepath, as_attachment=True, mimetype="audio/wav")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
