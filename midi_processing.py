import mido
import os
import subprocess
from constants import FRAME_DURATION

def modify_midi_file(midi_path, bpm, instrument):
    """
    Modifica un archivo MIDI dado cambiando el tempo y el instrumento.

    Args:
        midi_path (str): Ruta al archivo MIDI que se va a modificar.
        bpm (int): Nuevo tempo en beats por minuto (bpm).
        instrument (int): Nuevo número de instrumento MIDI para la pista.

    Returns:
        mido.MidiFile: Archivo MIDI modificado.
    
    Raises:
        ValueError: Si el archivo MIDI especificado no existe.
    """
    # Check if midi_path exists
    if not os.path.exists(midi_path):
        raise ValueError(f"File {midi_path} does not exist.")
    # Cargar el archivo MIDI
    midi_file = mido.MidiFile(midi_path)
    # Crear un nuevo archivo MIDI
    output_midi = mido.MidiFile()

    # Crear una nueva pista para el archivo MIDI de salida
    output_track = mido.MidiTrack()
    output_midi.tracks.append(output_track)
    # Agregar la nueva pista al archivo MIDI de salida
    output_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    output_track.append(mido.Message('program_change', program=instrument, channel=0, time=0))
    
    for msg in midi_file.tracks[0]:
        if msg.type != 'set_tempo' and msg.type != 'program_change':
            # Copiar otros mensajes MIDI
            output_track.append(msg)
    return output_midi

def pianoroll_to_midi(piano_roll, tempo=500000):
    """
    Convierte un piano roll binario en un archivo MIDI.

    Args:
        piano_roll (torch.Tensor): Piano roll como tensor binario.
        tempo (int, opcional): Tempo del archivo MIDI en microsegundos por beat.

    Returns:
        mido.MidiFile: Archivo MIDI generado.
    """
    piano_roll=piano_roll.T
    TICK_DURATION=(tempo)/(1_000_000*480)
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

def midi_to_wav(midi_path, wav_path):
    """
    Convierte un archivo MIDI a formato WAV utilizando Timidity++.

    Args:
        midi_path (str): Ruta al archivo MIDI de entrada.
        wav_path (str): Ruta al archivo WAV de salida.

    Returns:
        None
    """
    # Usar timidity para convertir MIDI a WAV
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path])