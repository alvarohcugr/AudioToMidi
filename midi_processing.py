import mido
import os
def modify_midi_file(midi_path, bpm, instrument):
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