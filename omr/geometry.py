"""
Módulo de utilidades geométricas para procesamiento de imágenes.
"""
import cv2
import numpy as np
from dataclasses import dataclass

# Tamaño "estándar" de trabajo (ancho, alto) en píxeles
# No tiene por qué ser el tamaño real del PDF, es sólo un lienzo de trabajo fijo.
A4_SIZE = (2480, 3508)

@dataclass
class RectRel:
    """
    Rectángulo en coordenadas relativas [0,1] respecto al ancho/alto de la imagen.
    x0, y0 = esquina superior izquierda
    x1, y1 = esquina inferior derecha
    """
    x0: float
    y0: float
    x1: float
    y1: float

def recortar_roi_rel(img_bgr, rect: RectRel):
    """
    Recorta un ROI de la imagen usando coordenadas relativas (0..1).
    Devuelve (roi_bgr, (x, y, w, h)) en coordenadas absolutas.
    """
    h, w = img_bgr.shape[:2]
    x0 = int(rect.x0 * w)
    y0 = int(rect.y0 * h)
    x1 = int(rect.x1 * w)
    y1 = int(rect.y1 * h)
    return img_bgr[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)

def detectar_hoja_y_enderezar(img_bgr):
    """
    Aquí está el core. El siguiente paso es pintar las esquinas.
    Intenta detectar la hoja como el contorno grande con forma aproximada de
    rectángulo A4. Si no lo consigue, hace fallback: escala toda la imagen a A4_SIZE.
    """
    h_img, w_img = img_bgr.shape[:2]
    img_area = w_img * h_img

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarizamos antes de Canny para intentar resaltar mejor la hoja
    _, thr = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thr, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: redimensionar toda la imagen
        print("No se han encontrado contornos.")
        return cv2.resize(img_bgr, A4_SIZE)

    best_cnt = None
    best_score = -1.0
    A4_RATIO = A4_SIZE[1] / A4_SIZE[0]  # alto/ancho ~ 1.41

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.2 * img_area:  # ignorar cosas pequeñas (<20% de la imagen)
            continue

        # Rectángulo mínimo que envuelve el contorno
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0:
            continue

        ratio = max(w, h) / min(w, h)

        # Queremos algo parecido a A4_RATIO, aunque no perfecto
        ratio_score = 1.0 - min(abs(ratio - A4_RATIO), 1.0)  # 1 → perfecto, 0 → muy mal
        area_score = area / img_area                           # 0..1 según lo grande que sea

        score = ratio_score * 0.6 + area_score * 0.4

        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        # Fallback: redimensionar toda la imagen
        return cv2.resize(img_bgr, A4_SIZE)

    # Obtenemos los 4 puntos del rectángulo mínimo
    rect = cv2.minAreaRect(best_cnt)
    box = cv2.boxPoints(rect)
    pts = np.array(box, dtype="float32")

    # Ordenamos los vértices como antes (top-left, top-right, bottom-right, bottom-left)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    w, h = A4_SIZE
    dst = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h))
    return warped
