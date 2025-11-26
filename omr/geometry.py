
"""
Funciones de geometría para el OMR:
- Detectar la hoja de examen en la imagen de entrada.
- Enderezarla y recortarla a un A4 "ideal" (test_case_A4).

NO se definen aquí ROIs ni nada relacionado con QR o burbujas.
"""

import cv2
import numpy as np

# Tamaño de trabajo para la hoja enderezada (A4 vertical en píxeles)
A4_WIDTH, A4_HEIGHT = 2480, 3508
A4_SIZE = (A4_WIDTH, A4_HEIGHT)


def enderezar_hoja(img_bgr):
    """
    Detecta la hoja de examen (folio blanco sobre fondo oscuro),
    calcula sus 4 esquinas y aplica una transformación de perspectiva
    para obtener una imagen A4 "plana" y recortada.

    Parámetros
    ----------
    img_bgr : np.ndarray
        Imagen original en BGR (OpenCV), puede contener mesa, teclado, etc.

    Devuelve
    --------
    hoja_warped : np.ndarray
        Imagen BGR de la hoja enderezada, tamaño fijo A4_SIZE.
    """

    h_img, w_img = img_bgr.shape[:2]
    img_area = w_img * h_img

    # 1) Pasamos a escala de grises y suavizamos
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Umbralización tipo Otsu: intentamos separar hoja (blanca) y fondo
    _, thr = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Si el borde de la imagen es mayoritariamente blanco, invertimos
    # (caso en el que la hoja salga negra y el fondo claro).
    border = np.concatenate([
        thr[0, :], thr[-1, :], thr[:, 0], thr[:, -1]
    ])
    if border.mean() > 200:  # bordes muy blancos
        thr = cv2.bitwise_not(thr)

    # 3) Contornos externos: buscamos la hoja como el contorno más grande
    contours, _ = cv2.findContours(
        thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Si no encontramos nada, devolvemos la imagen reescalada para no romper flujo
        return cv2.resize(img_bgr, A4_SIZE)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # Si el mayor contorno es muy pequeño en comparación con la imagen, algo va mal
    if area < 0.3 * img_area:
        return cv2.resize(img_bgr, A4_SIZE)

    # 4) Aproximamos el contorno a un polígono y tratamos de obtener 4 puntos
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        # Si no son exactamente 4 vértices, usamos el rectángulo mínimo
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = np.array(box, dtype="float32")

    # 5) Ordenar los puntos como: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    # 6) Definimos el rectángulo destino (A4 ideal)
    w_dest, h_dest = A4_SIZE
    dst = np.array([
        [0,         0],
        [w_dest-1,  0],
        [w_dest-1,  h_dest-1],
        [0,         h_dest-1]
    ], dtype="float32")

    # 7) Matriz de transformación y warp
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w_dest, h_dest))

    return warped


# Alias para mantener compatibilidad con el nombre anterior si ya lo usabas
def detectar_hoja_y_enderezar(img_bgr):
    """
    Alias de enderezar_hoja para compatibilidad.
    """
    return enderezar_hoja(img_bgr)
