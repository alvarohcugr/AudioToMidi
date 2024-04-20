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
          console.log("Hola")
          file = event.dataTransfer.items[0].getAsFile();
      }
  } else {
      // Si no hay items, toma el primer archivo del array de archivos
      if (event.dataTransfer.files.length > 0) {
        console.log("Hola2")
          file = event.dataTransfer.files[0];
      }
  }

  if (file) {
      var input = document.getElementById('audio_file');
      var fileNameSpan = document.getElementById('file-name');

      var dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      input.files = dataTransfer.files; // Asigna los archivos al input file

      fileNameSpan.textContent = file.name; // Actualiza el texto del span con el nombre del archivo
  }
}

function handleDragOver(event) {
  event.preventDefault();
}

function handleDragEnter(event) {
  event.preventDefault();
  event.target.classList.add('highlight');
}

function handleDragLeave(event) {
  event.preventDefault();
  event.target.classList.remove('highlight');
}


var dropZone = document.getElementById('drop_zone');
dropZone.addEventListener('drop', dropHandler, false);
dropZone.addEventListener('dragover', handleDragOver, false);
dropZone.addEventListener('dragenter', handleDragEnter, false);
dropZone.addEventListener('dragleave', handleDragLeave, false);