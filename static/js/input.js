const fileNameSpan = document.getElementById("file-name");
const convertButton = document.getElementById("convert_button");

function validateAudio(file){
  var valid=file.name.endsWith('.wav') || file.name.endsWith('.mp3');
  if (!valid) {
    alert('Por favor, sube un archivo de tipo .wav o .mp3.');
  }
  return valid;
}
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

function handleDragOver(event) {
  event.preventDefault();
}

function handleDragEnter(event) {
  event.preventDefault();
  event.currentTarget.classList.add('highlight');
}

function handleDragLeave(event) {
  event.preventDefault();
  // Verificar si el puntero del ratón ha salido del drop_zone o si ha entrado en otro elemento interno
  if (!event.relatedTarget || (event.relatedTarget !== this && !this.contains(event.relatedTarget))) {
    this.classList.remove('highlight');
  }
}


var dropZone = document.getElementById('drop_zone');
dropZone.addEventListener('drop', dropHandler, false);
dropZone.addEventListener('dragover', handleDragOver, false);
dropZone.addEventListener('dragenter', handleDragEnter, false);
dropZone.addEventListener('dragleave', handleDragLeave, false);

document.getElementById("upload-form").addEventListener("submit", function() {
  document.getElementById("loader-convert").style.display = "block";
  convertButton.disabled = true;
});
// Solo se ejecutará cuando el usuario haya hecho la conversión
var tempoInput = document.getElementById("tempo");
var tempoValueSpan = document.getElementById("tempo-value");
if (tempoInput && tempoValueSpan) {
  tempoInput.addEventListener("input", function() {
  tempoValueSpan.textContent = "x" + Math.round(this.value / 120.0 * 100.0) / 100.0;
  });
}
var modifyForm= document.getElementById("modify-form");
if (modifyForm) {
  modifyForm.addEventListener("submit", function() {
    document.getElementById("loader-modify").style.display = "block";
    document.getElementById("modify_button").disabled = true;
  });
}