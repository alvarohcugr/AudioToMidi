import torch
import mido
import numpy as np
import librosa


def predict(model, inputs):
  # Hacer una predicción con el modelo entrenado
  with torch.no_grad():
      prediction=np.array([model(torch.from_numpy(input)) for input in inputs])

  # Convertir las predicciones a una forma interpretable
  # (Dependiendo de la salida de tu modelo, esto puede variar)
  prediction = (prediction.T > 0.5)  # Ejemplo: binarización para una salida sigmoide
  return prediction

def pianoroll_to_midi(piano_roll, tempo=439440, hop_length=36):
    # Crear archivo midi 
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(mido.Message('program_change', program=0))  # Cambio de programa (instrumento)

    last_frame=piano_roll[0]
    last_time=0
    # Añadir los mensajes del primer frame
    for i_note in range(0, last_frame.shape[0]):
      if (piano_roll[0][i_note]):
        track.append(mido.Message('note_on', note=i_note + 21, velocity=100, time=(i_frame-last_time)*hop_length))
    # Añadir los mensajes del resto de frames
    for i_frame in range(1, piano_roll.shape[0]):
      for i_note in range(0, last_frame.shape[0]):
        if (last_frame[i_note] and not piano_roll[i_frame][i_note]):
          track.append(mido.Message('note_off', note=i_note + 21, velocity=0, time=(i_frame-last_time)*hop_length))
          last_time=i_frame
        elif (not last_frame[i_note] and piano_roll[i_frame][i_note]):
          track.append(mido.Message('note_on', note=i_note + 21, velocity=100, time=(i_frame-last_time)*hop_length))
          last_time=i_frame
      last_frame=piano_roll[i_frame]
    return mid_new

def get_wav(path, sr=16000):
  # Cargar el archivo .wav y obtener la señal y la tasa de muestreo
  return librosa.load(path, sr=sr)

def get_cqt(y, sr):
  # Calcular el Constant-Q Transform (CQT)
  cqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=36, n_bins=252, fmin=27.5))

  return cqt

def get_features(wav_file_path):
    y,sr=get_wav(wav_file_path)
    return get_cqt(y, sr)

def get_midi_from_wav(wav_file_path, model):
    # Obtener características
    features = get_features(wav_file_path)

    # Normalizar las características
    features = (features - 0.014933423) / 0.048849646
    # Predecir
    prediction = predict(model, features.T)

    # Convertir a formato MIDI
    midi_file = pianoroll_to_midi(prediction.T)

    return midi_file