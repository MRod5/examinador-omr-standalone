# main.py
import cv2
from omr.capture import cargar_imagen_desde_fichero
from omr.geometry import enderezar_hoja
from omr.rois import dibujar_qr_area

# Cambia esta ruta a tu imagen real (HEIC/JPG/PNG)
RUTA_IMAGEN = "imagenes/pruebas_OMR.HEIC"   # o "imagenes/test_omr_iphone.jpg"

def main():
    # 1) Cargar imagen original desde fichero
    frame = cargar_imagen_desde_fichero(RUTA_IMAGEN)
    if frame is None:
        print("No se pudo cargar la imagen.")
        return

    # 2) Enderezar y recortar la hoja (test_case_A4)
    hoja = enderezar_hoja(frame)

    # 3) Dibujar la caja verde del QR sobre la hoja enderezada
    hoja_debug = hoja.copy()
    dibujar_qr_area(hoja_debug)

    # 4) Mostrar resultado
    cv2.imshow("Hoja enderezada con caja QR (verde)", hoja_debug)
    print("Pulsa cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
