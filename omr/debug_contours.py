"""
Módulo para depurar la detección de contornos con OpenCV.
"""
import cv2

def mostrar_contornos(frame_bgr):
    """
    Muestra:
      - Imagen original
      - Imagen binarizada (threshold)
      - Imagen con contornos dibujados
    para entender qué está detectando OpenCV.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarizamos con Otsu
    _, thr = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Bordes para encontrar contornos
    edges = cv2.Canny(thr, 50, 150)

    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contornos encontrados: {len(contours)}")

    # Dibujamos TODOS los contornos en verde
    contornos_img = frame_bgr.copy()
    cv2.drawContours(contornos_img, contours, -1, (0, 255, 0), 1)

    # Mostramos todo
    cv2.imshow("Original", frame_bgr)
    cv2.imshow("Threshold", thr)
    cv2.imshow("Contornos (en verde)", contornos_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
