const fileNameSpan = document.getElementById("file-name");
const convertButton = document.getElementById("convert_button");

/**
 * Valida si el archivo subido es un archivo de audio de tipo .wav o .mp3.
 * @param {File} file - El archivo que se va a validar.
 * @returns {boolean} - Devuelve true si el archivo es válido, de lo contrario false.
 */
function validateAudio(file){
  var valid=file.name.endsWith('.wav') || file.name.endsWith('.mp3');
  if (!valid) {
    alert('Por favor, sube un archivo de tipo .wav o .mp3.');
  }
  return valid;
}
/**
 * Maneja la selección de un archivo mediante el input.
 * @param {HTMLInputElement} input - El input de archivo.
 */
function selectFileHandler(input) {
    if (input.files.length > 0) {
        if (validateAudio(input.files[0])) {
            fileNameSpan.textContent = input.files[0].name;
            convertButton.disabled = false;
            convertButton.classList.remove("disabled");
        }
        else {
            fileNameSpan.textContent = "Ningún archivo seleccionado";
            convertButton.disabled = true;
            convertButton.classList.add("disabled");
        }
  }
}
/**
 * Maneja el evento de soltar un archivo en la zona de drop.
 * @param {DragEvent} event - El evento de drop.
 */
function dropHandler(event) {
  event.preventDefault();

  var file;

  if (event.dataTransfer.items) {
      // Si hay más de un archivo, toma el primero
      if (event.dataTransfer.items.length > 0 && event.dataTransfer.items[0].kind === 'file') {
          file = event.dataTransfer.items[0].getAsFile();
      }
  } else {
      // Si no hay items, toma el primer archivo del array de archivos
      if (event.dataTransfer.files.length > 0) {
          file = event.dataTransfer.files[0];
      }
  }

  if (file) {
    // Verificar si el archivo de audio es válido
    if (validateAudio(file)) {
        convertButton.disabled = false;
        convertButton.classList.remove('disabled');
        var input = document.getElementById('audio_file');
        var dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;

        fileNameSpan.textContent = file.name;
    } else {
        fileNameSpan.textContent = "Ningún archivo seleccionado"; 
        convertButton.disabled = true;
        convertButton.classList.add('disabled');
    }
  }
  event.currentTarget.classList.remove('highlight');
}

/**
 * Maneja el evento de arrastrar sobre la zona de drop.
 * @param {DragEvent} event - El evento de arrastre.
 */
function handleDragOver(event) {
  event.preventDefault();
}


/**
 * Maneja el evento de entrar en la zona de drop.
 * @param {DragEvent} event - El evento de entrada.
 */
function handleDragEnter(event) {
  event.preventDefault();
  event.currentTarget.classList.add('highlight');
}

/**
 * Maneja el evento de salir de la zona de drop.
 * @param {DragEvent} event - El evento de salida.
 */
function handleDragLeave(event) {
  event.preventDefault();
  // Verificar si el puntero del ratón ha salido del drop_zone o si ha entrado en otro elemento interno
  if (!event.relatedTarget || (event.relatedTarget !== this && !this.contains(event.relatedTarget))) {
    this.classList.remove('highlight');
  }
}

// Añadir los eventos de drag and drop a la zona de drop
var dropZone = document.getElementById('drop_zone');
dropZone.addEventListener('drop', dropHandler, false);
dropZone.addEventListener('dragover', handleDragOver, false);
dropZone.addEventListener('dragenter', handleDragEnter, false);
dropZone.addEventListener('dragleave', handleDragLeave, false);
// Maneja el evento de envío del formulario de subida
document.getElementById("upload-form").addEventListener("submit", function() {
  document.getElementById("loader-convert").style.display = "block";
  convertButton.disabled = true;
});

// Actualiza la visualización del valor del tempo
var tempoInput = document.getElementById("tempo");
var tempoValueSpan = document.getElementById("tempo-value");
if (tempoInput && tempoValueSpan) {
  tempoValueSpan.textContent = "x" + Math.round(tempoInput.value / 120.0 * 100.0) / 100.0 + "\t(" + tempoInput.value + " MIDI BPM)";
  tempoInput.addEventListener("input", function() {
  tempoValueSpan.textContent = "x" + Math.round(this.value / 120.0 * 100.0) / 100.0 + "\t(" + tempoInput.value + " MIDI BPM)";
  });
}

// Maneja el evento de envío del formulario de modificación
var modifyForm= document.getElementById("modify-form");
if (modifyForm) {
  modifyForm.addEventListener("submit", function() {
    document.getElementById("loader-modify").style.display = "block";
    document.getElementById("modify_button").disabled = true;
  });
}