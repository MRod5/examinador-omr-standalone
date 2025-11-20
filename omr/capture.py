"""
Módulo para capturar un frame desde la webcam.
"""
import cv2

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
