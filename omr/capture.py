"""
Módulo para capturar un frame desde la webcam o desde fichero.
"""

import cv2
import os
from PIL import Image
import pillow_heif
import numpy as np

def capturar_frame_desde_webcam(camera_index: int = 0):
    """
    Abre la cámara, muestra la imagen en vivo y devuelve un frame
    cuando el usuario pulsa ESPACIO. Si pulsa ESC, devuelve None.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")

    frame_final = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        cv2.imshow("Coloca la hoja y pulsa ESPACIO para capturar (ESC para cancelar)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Espacio
            frame_final = frame.copy()
            break
        if key == 27:  # Escape
            frame_final = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame_final



def cargar_imagen_desde_fichero(ruta):
    """
    Carga HEIC/JPG/PNG como imagen BGR (cv2).
    """

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encuentra el fichero: {ruta}")

    # Si es HEIC → convertir a imagen PIL primero
    extension = ruta.lower().split(".")[-1]
    if extension == "heic":
        heif_file = pillow_heif.read_heif(ruta)
        img_pil = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data
        )
        img = cv2.cvtColor(
            cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2RGB
        )
    else:
        img = cv2.imread(ruta)

    if img is None:
        raise ValueError(f"No se ha podido leer la imagen: {ruta}")

    # Si está rotada horizontalmente, la enderezamos a vertical
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img
