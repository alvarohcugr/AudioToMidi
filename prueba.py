import mido
from mido import MidiFile
from mido.backends import portmidi

def list_ports():
    print("Puertos de salida MIDI disponibles:")
    for i, port in enumerate(mido.get_output_names()):
        print(f"{i}: {port}")

def play_midi_file(filename, port_index):
    mid = MidiFile(filename)
    
    with mido.open_output(mido.get_output_names()[port_index]) as output:
        for message in mid.play():
            output.send(message)

if __name__ == "__main__":
    filename = "/home/alvaro/Descargas/prediccion.mid"  # Cambia esto por la ruta de tu archivo MIDI
    list_ports()
    
    port_index = int(input("Selecciona el número del puerto de salida MIDI: "))
    
    play_midi_file(filename, port_index)