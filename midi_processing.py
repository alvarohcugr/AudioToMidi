import mido
def modify_midi_file(midi_path, tempo, instrument):
    # Cargar el archivo MIDI
    midi_file = mido.MidiFile(midi_path)

    # Crear un nuevo archivo MIDI
    output_midi = mido.MidiFile()

    # Crear una nueva pista para el archivo MIDI de salida
    output_track = mido.MidiTrack()
    output_midi.tracks.append(output_track)

    for msg in midi_file:
        # Copiar mensajes de metaevento excepto el de tempo
        if msg.type == 'set_tempo':
            # Calcular el nuevo valor de tempo
            microseconds_per_beat = mido.bpm2tempo(tempo)
            msg = mido.MetaMessage('set_tempo', tempo=microseconds_per_beat, time=int(msg.time))
        # Copiar mensajes de control de cambio de instrumento
        elif msg.type == 'program_change':
            msg = mido.Message('program_change', program=instrument, channel=msg.channel, time=int(msg.time))
        # Copiar otros mensajes MIDI
        output_track.append(msg)

    # Guardar el archivo MIDI modificado
    output_midi.save(midi_path)

    return midi_path