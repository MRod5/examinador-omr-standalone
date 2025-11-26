from omr.capture import capturar_frame_desde_webcam
from omr.geometry import detectar_hoja_y_enderezar
from omr.rois_debug import dibujar_rois_y_malla

def main():
    frame = capturar_frame_desde_webcam()
    if frame is None:
        print("Captura cancelada.")
        return

    hoja = detectar_hoja_y_enderezar(frame)
    dibujar_rois_y_malla(hoja)

if __name__ == "__main__":
    main()
