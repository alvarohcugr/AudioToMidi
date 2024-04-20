function updateFileName(input) {
  const fileName = input.files[0]?.name || 'Ningún archivo seleccionado';
  document.getElementById('file-name').textContent = fileName;
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
    // Verificar si el archivo es de tipo .wav
    if (file.type === 'audio/x-wav' || file.type === 'audio/wav') {
        var input = document.getElementById('audio_file');
        var fileNameSpan = document.getElementById('file-name');

        var dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;

        fileNameSpan.textContent = file.name;
    } else {
      console.log(file.type);
        alert('Por favor, sube un archivo de tipo .wav.');
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
  /* event.currentTarget.classList.remove('highlight'); */
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