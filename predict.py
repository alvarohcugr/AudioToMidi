import torch
import mido
import numpy as np
import librosa
from constants import FRAME_DURATION, N_BINS_PER_OCTAVE, N_BINS, F_MIN, SAMPLING_RATE, HOP_LENGTH
import matplotlib.pyplot as plt
from midi_processing import pianoroll_to_midi


def post_process_outputs(output_notes, output_onsets=None, tolerance=2, onset_thres=0.5, note_thres=0.3, add_note_thres=0.5):
    """
    Procesa las salidas del modelo para generar eventos de notas basados en umbrales de probabilidad.

    Args:
        output_notes (torch.Tensor): Tensor con las probabilidades de notas predichas.
        output_onsets (torch.Tensor, opcional): Tensor con las probabilidades de onsets predichas.
        tolerance (int, opcional): Tolerancia de duración mínima de una nota.
        onset_thres (float, opcional): Umbral de probabilidad para onsets.
        note_thres (float, opcional): Umbral de probabilidad para fin de notas.
        add_note_thres (float, opcional): Umbral de probabilidad adicional para notas.

    Returns:
        list: Lista de eventos de notas con tiempos de inicio, fin y tono.
    """
    note_events = []
    if (output_onsets != None):
        # Tomar como candidatos peak onsets con probabilidad > 0.5
        # Calcula la matriz de diferencias para detectar los picos
        diff_tensor = torch.diff(output_onsets, dim=1)

        # Encuentra las posiciones donde la diferencia cambia de positivo a negativo
        positive_to_negative = torch.nonzero((diff_tensor[:, :-1] > 0) & (diff_tensor[:, 1:] < 0))

        # Ajusta las posiciones para considerar el desplazamiento por torch.diff
        onset_candidates = list(zip(positive_to_negative[:, 0], positive_to_negative[:, 1] + 1))

        # Filtra los candidatos según el umbral
        onset_candidates = [(i[0].item(), i[1].item()) for i in onset_candidates if output_onsets[i[0].item(), i[1].item()] >= onset_thres]

        # Crear eventos de notas a partir de los candidatos y las probabilidades de notas
        for onset in onset_candidates[::-1]:  # Iterar en orden temporal inverso
            start_time = onset[1]
            pitch = onset[0]
            end_time = start_time
            under_thres=0
            for t in range(start_time + 1, output_notes.shape[1]):
                if (output_notes[pitch,t]<note_thres):
                    under_thres+=1
                else:
                    under_thres=0
                    end_time=t
                if (under_thres>=tolerance):
                    break
            # No tener en cuenta la nota si es muy corta
            note_duration=(end_time+1-start_time)*FRAME_DURATION

            output_notes[pitch, start_time:end_time + 1] = 0
            note_events.append({"start_time":start_time, "end_time":end_time, "pitch": pitch})
    # Crear eventos de notas adicionarles a partir de las probabilidades de notas
    for pitch in range(output_notes.shape[0]):
        for frame in range(output_notes.shape[1]):
            likelihood=output_notes[pitch, frame]
            if (likelihood > add_note_thres):
                start_time=frame
                end_time=frame
                under_thres=0
                for t in range(frame + 1, output_notes.shape[1]):
                    if (output_notes[pitch,t]<note_thres):
                        under_thres+=1
                    else:
                        under_thres=0
                        end_time=t
                    if (under_thres>=tolerance):
                        break
                under_thres=0
                for t in range(frame - 1, 0, -1):
                    if (output_notes[pitch,t]<note_thres):
                        under_thres+=1
                    else:
                        under_thres=0
                        start_time=t
                    if (under_thres>=tolerance):
                        break
                # No tener en cuenta la nota si es muy corta
                note_duration=(end_time+1-start_time)*FRAME_DURATION
                output_notes[pitch, start_time:end_time + 1] = 0
                note_events.append({"start_time":start_time, "end_time":end_time, "pitch": pitch})

    return note_events

def note_events_to_piano_roll(note_events, n_bins, n_frames):
    """
    Convierte eventos de notas en un piano roll (matriz binaria).

    Args:
        note_events (list): Lista de eventos de notas con inicio, fin y tono.
        n_bins (int): Número de bins (notas) en el piano roll.
        n_frames (int): Número de frames (frames) en el piano roll.

    Returns:
        torch.Tensor: Piano roll como tensor binario.
    """
    piano_roll = torch.zeros((n_bins, n_frames), dtype=torch.float32)
    for note_event in note_events:
        start_time = note_event['start_time']
        end_time = note_event['end_time']
        pitch = note_event['pitch']
        piano_roll[pitch, start_time:end_time + 1] = 1
    return piano_roll
def predict(model, inputs, onsets=False):
    """
    Realiza una predicción utilizando las salidas de notas y, opcionalmente, onsets.

    Args:
        model (torch.nn.Module): Modelo de red neuronal.
        inputs (torch.Tensor): Tensor de entrada para la predicción.
        onsets (bool, opcional): Indica si el modelo produce predicciones de onsets.

    Returns:
        torch.Tensor: Piano roll predicho como tensor binario.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        cqt = torch.tensor(inputs).to(device)
        # Añadir dimension de batch
        cqt=torch.unsqueeze(cqt, 0)
        if (onsets):
            # Obtener espectrograma pasar como input al modelo
            output_notes, output_onsets = model(cqt)
            # Eliminar la dimensión de los canales
            output_notes=torch.squeeze(output_notes)
            output_onsets=torch.squeeze(output_onsets)
            # Aplicar sigmoide para obtener predicción binaria
            output_notes=torch.sigmoid(output_notes)
            output_onsets=torch.sigmoid(output_onsets)
            # Aplicar el post-procesado a las salidas del modelo
            note_events=post_process_outputs(output_notes, output_onsets)
        else:
            output_notes = model(cqt)
            # Eliminar la dimensión de los canales
            output_notes=torch.squeeze(output_notes)
            # Aplicar sigmoide para obtener predicción binaria
            output_notes=torch.sigmoid(output_notes)
            # Aplicar el post-procesado a las salidas del modelo
            note_events=post_process_outputs(output_notes)
        # Obtener la predicción en forma de piano roll
        predicted_pianoroll=note_events_to_piano_roll(note_events, output_notes.shape[0], output_notes.shape[1])
    return predicted_pianoroll

def get_wav(path, sr=SAMPLING_RATE):
    """
        Carga un archivo WAV y devuelve la señal de audio y la tasa de muestreo.

        Args:
            path (str): Ruta al archivo WAV.
            sr (float, opcional): Tasa de muestreo deseada.

        Returns:
            np.ndarray, float: Señal de audio y tasa de muestreo.
        """
    # Cargar el archivo .wav y obtener la señal y la tasa de muestreo
    return librosa.load(path, sr=sr)
def get_cqt(y, sr):

    """
        Calcula el Constant-Q Transform (CQT) de una señal de audio.

        Args:
            y (np.ndarray): Señal de audio.
            sr (float): Tasa de muestreo de la señal de audio.

        Returns:
            np.ndarray: CQT calculado como matriz de magnitudes.
        """
  # Calcular el Constant-Q Transform (CQT)
    cqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=N_BINS_PER_OCTAVE, n_bins=N_BINS, fmin=F_MIN, hop_length=HOP_LENGTH))
    return cqt

def visualize_piano_roll(piano_roll, name, cmap='Blues', title='Piano-Roll'):
    """
    Visualiza un piano roll como imagen y la guarda en un archivo.

    Args:
        piano_roll (torch.Tensor): Piano roll como tensor binario.
        name (str): Nombre del archivo de imagen a guardar.
        cmap (str, opcional): Mapa de colores para la visualización.
        title (str, opcional): Título de la imagen.

    Returns:
        None
    """
    plt.figure(figsize=(16, 4))
    plt.imshow(piano_roll, aspect='auto', cmap=cmap, vmin=0, vmax=1, origin='lower', extent=[0,piano_roll.shape[1], 0,  piano_roll.shape[0]], interpolation="none")
    plt.xlabel('Tiempo')
    plt.ylabel('Nota')
    plt.title(title)
    plt.savefig(name) # Guardar la imagen()

def get_midi_from_wav(wav_file_path, model, onsets=False):
    """
    Convierte un archivo WAV en un archivo MIDI utilizando un modelo de red neuronal.

    Args:
        wav_file_path (str): Ruta al archivo WAV.
        model (torch.nn.Module): Modelo de red neuronal entrenado.
        onsets (bool, opcional): Indica si el modelo predice los onsets.

    Returns:
        mido.MidiFile: Archivo MIDI generado.
    """
    y,sr=get_wav(wav_file_path)
    # Obtener características
    features = get_cqt(y, sr)
    # Normalizar las características
    features = (np.log(features+10e-7) - (-6.3362346)) / 2.29297
    # Predecir
    prediction = predict(model, features, onsets=onsets)
    # Convertir a formato MIDI
    midi_file = pianoroll_to_midi(prediction)

    return midi_file