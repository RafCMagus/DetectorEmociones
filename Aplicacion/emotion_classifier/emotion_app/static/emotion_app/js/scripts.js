document.addEventListener("DOMContentLoaded", function () {
    let galleryButton = document.getElementById("show-gallery");
    let galleryModal = document.getElementById("gallery-modal");
    let closeGallery = document.querySelector(".close-gallery");
    let galleryContainer = document.getElementById("gallery-container");

    galleryButton.addEventListener("click", function () {
        fetch("D:/ModeloIA/Aplicacion/emotion_classifier/output/")  // Endpoint en Django para obtener imágenes
            .then(response => response.json())
            .then(data => {
                galleryContainer.innerHTML = ""; // Limpiar la galería
                if (data.images.length > 0) {
                    data.images.forEach(image => {
                        let imgElement = document.createElement("img");
                        imgElement.src = "/static/output/" + image;
                        imgElement.classList.add("gallery-image");
                        galleryContainer.appendChild(imgElement);
                    });
                    galleryModal.style.display = "block"; // Mostrar la galería
                } else {
                    alert("No hay imágenes procesadas.");
                }
            })
            .catch(error => console.error("Error al obtener imágenes:", error));
    });

    closeGallery.addEventListener("click", function () {
        galleryModal.style.display = "none"; // Cerrar la galería
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            galleryModal.style.display = "none"; // Cerrar con la tecla ESC
        }
    });
});
