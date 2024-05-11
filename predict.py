import torch
import mido
import numpy as np
import librosa
from constants import FRAME_DURATION, N_BINS_PER_OCTAVE, N_BINS, F_MIN, SAMPLING_RATE, HOP_LENGTH
import matplotlib.pyplot as plt


def post_process_outputs(output_notes, output_onset, tolerance=4, onset_thres=0.5, note_thres=0.3, add_note_thres=0.5, note_duration_thres=0.12):
    # Tomar como candidatos peak onsets con probabilidad > 0.5
    # Calcula la matriz de diferencias para detectar los picos
    diff_tensor = torch.diff(output_onset, dim=1)

    # Encuentra las posiciones donde la diferencia cambia de positivo a negativo
    positive_to_negative = torch.nonzero((diff_tensor[:, :-1] > 0) & (diff_tensor[:, 1:] < 0))

    # Ajusta las posiciones para considerar el desplazamiento por torch.diff
    onset_candidates = list(zip(positive_to_negative[:, 0], positive_to_negative[:, 1] + 1))

    # Filtra los candidatos según el umbral
    onset_candidates = [(i[0].item(), i[1].item()) for i in onset_candidates if output_onset[i[0].item(), i[1].item()] >= onset_thres]

    # Crear eventos de notas a partir de los candidatos y las probabilidades de notas
    note_events = []
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
        if note_duration < note_duration_thres:
            continue

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
                if note_duration < note_duration_thres:
                    continue
                note_events.append({"start_time":start_time, "end_time":end_time, "pitch": pitch})

    return note_events


def note_events_to_piano_roll(note_events, n_bins, n_frames):
    piano_roll = torch.zeros((n_bins, n_frames), dtype=torch.float32)
    for note_event in note_events:
        start_time = note_event['start_time']
        end_time = note_event['end_time']
        pitch = note_event['pitch']
        piano_roll[pitch, start_time:end_time + 1] = 1
    return piano_roll
def predict(model, inputs):
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.eval()
  with torch.no_grad():
      # Obtener espectrograma pasar como input al modelo
      cqt= torch.tensor(inputs).to(device)

      # Añadir dimension de batch
      cqt=torch.unsqueeze(cqt, 0)

      output_notes, output_onsets = model(cqt)
      # Eliminar la dimensión de los canales y de batch
      output_notes=torch.squeeze(output_notes)
      output_onsets=torch.squeeze(output_onsets)

      # Aplicar sigmoide para obtener predicción binaria
      output_notes=torch.sigmoid(output_notes)
      output_onsets=torch.sigmoid(output_onsets)

      visualize_piano_roll(output_notes, name='output_notes.png')
      visualize_piano_roll(output_onsets, name='output_onsets.png')

      note_events=post_process_outputs(output_notes, output_onsets)

      predicted_pianoroll=note_events_to_piano_roll(note_events, output_notes.shape[0], output_notes.shape[1])
  return predicted_pianoroll

def pianoroll_to_midi(piano_roll, tempo=500000):
    piano_roll=piano_roll.T
    bpm=60_000_000/tempo
    TICK_DURATION=60/(bpm*480)
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
        track.append(mido.Message('note_on', note=i_note + 21, velocity=100, time=0))
    # Añadir los mensajes del resto de frames
    for i_frame in range(1, piano_roll.shape[0]):
      for i_note in range(0, last_frame.shape[0]):
        if (last_frame[i_note] and not piano_roll[i_frame][i_note]):
          tick_time=int((i_frame-last_time)*FRAME_DURATION/TICK_DURATION+1)
          track.append(mido.Message('note_off', note=i_note + 21, velocity=0, time=tick_time))
          last_time=i_frame
        elif (not last_frame[i_note] and piano_roll[i_frame][i_note]):
          tick_time=int((i_frame-last_time)*FRAME_DURATION/TICK_DURATION+1)
          track.append(mido.Message('note_on', note=i_note + 21, velocity=100, time=tick_time))
          last_time=i_frame
      last_frame=piano_roll[i_frame]
    return mid_new

def get_wav(path, sr=SAMPLING_RATE):
  # Cargar el archivo .wav y obtener la señal y la tasa de muestreo
  return librosa.load(path, sr=sr)

def get_cqt(y, sr):
  # Calcular el Constant-Q Transform (CQT)
  cqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=N_BINS_PER_OCTAVE, n_bins=N_BINS, fmin=F_MIN, hop_length=HOP_LENGTH))
  return cqt

def get_features(wav_file_path):
    y,sr=get_wav(wav_file_path)
    return get_cqt(y, sr)
def visualize_piano_roll(piano_roll, name, cmap='Blues', title='Piano-Roll'):
    plt.figure(figsize=(16, 4))
    plt.imshow(piano_roll, aspect='auto', cmap=cmap, vmin=0, vmax=1, origin='lower', extent=[0,piano_roll.shape[1], 0,  piano_roll.shape[0]], interpolation="none")
    plt.xlabel('Tiempo')
    plt.ylabel('Nota')
    plt.title(title)
    plt.savefig(name) # Guardar la imagen()

def get_midi_from_wav(wav_file_path, model):
    # Obtener características
    features = get_features(wav_file_path)
    # Normalizar las características
    features = (np.log(features+10e-7) - (-6.3362346)) / 2.29297
    # Predecir
    prediction = predict(model, features)
    # Convertir a formato MIDI
    midi_file = pianoroll_to_midi(prediction)

    return midi_file