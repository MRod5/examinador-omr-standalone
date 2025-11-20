"""
Módulo principal para capturar un frame desde la webcam y mostrarlo.
"""

from omr.capture import capturar_frame_desde_webcam
import cv2

def main():
    frame = capturar_frame_desde_webcam()
    if frame is None:
        print("Captura cancelada.")
        return

    # Mostramos la imagen capturada en una ventana aparte
    cv2.imshow("Frame capturado :)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
