"""
Módulo principal para capturar un frame desde la webcam y mostrarlo.
"""

from omr.capture import capturar_frame_desde_webcam
from omr.geometry import detectar_hoja_y_enderezar
import cv2

def main():
    frame = capturar_frame_desde_webcam()
    if frame is None:
        print("Captura cancelada.")
        return

    try:
        # Detecta e intenta enderezar la hoja A4 en el frame capturado
        hoja = detectar_hoja_y_enderezar(frame)
    except RuntimeError as e:
        print("Error al detectar la hoja:", e)
        # Mostramos al menos el frame original para diagnóstico
        cv2.imshow("Frame original", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Se muestra la hoja capturada y enderezada
    cv2.imshow("Hoja enderezada", hoja)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
