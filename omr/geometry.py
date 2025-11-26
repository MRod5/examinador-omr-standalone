# omr/geometry.py
import cv2
import numpy as np
from dataclasses import dataclass

# Tamaño de trabajo fijo (A4 vertical)
A4_SIZE = (2480, 3508)  # (ancho, alto)


@dataclass
class RectRel:
    x0: float
    y0: float
    x1: float
    y1: float


def recortar_roi_rel(img_bgr, rect: RectRel):
    h, w = img_bgr.shape[:2]
    x0 = int(rect.x0 * w)
    y0 = int(rect.y0 * h)
    x1 = int(rect.x1 * w)
    y1 = int(rect.y1 * h)
    return img_bgr[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


# ---- ROIs (ajustaremos si hace falta, pero dejamos estos de momento) ----

QR_AREA = RectRel(
    x0=0.79,
    y0=0.03,
    x1=0.94,
    y1=0.15
)

BUBBLES_AREA = RectRel(
    x0=0.065,
    y0=0.22,
    x1=0.935,
    y1=0.82
)

LEFT_COL = RectRel(
    x0=0.00, y0=0.00,
    x1=0.48, y1=1.00
)

RIGHT_COL = RectRel(
    x0=0.52, y0=0.00,
    x1=1.00, y1=1.00
)


def detectar_hoja_y_enderezar(img_bgr):
    """
    1) Detecta el contorno de la hoja (la región blanca más grande).
    2) Obtiene sus 4 esquinas.
    3) Warpea a un A4 perfecto (A4_SIZE).
    """

    h_img, w_img = img_bgr.shape[:2]
    img_area = w_img * h_img

    # --- 1. Escala de grises y suavizado ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. Umbral global tipo Otsu: hoja blanca, fondo negro ---
    _, thr = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Si por lo que sea la hoja saliera negra y el fondo blanco, invertimos
    # (comprobamos cuánta "tinta blanca" hay en los bordes).
    border = np.concatenate([
        thr[0, :], thr[-1, :], thr[:, 0], thr[:, -1]
    ])
    if border.mean() > 200:  # bordes muy blancos -> invertimos
        thr = cv2.bitwise_not(thr)

    # --- 3. Buscamos contornos EXTERNOS (la hoja) ---
    contours, _ = cv2.findContours(
        thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Fallback: si algo va muy mal, reescalamos toda la imagen
        return cv2.resize(img_bgr, A4_SIZE)

    # Contorno más grande -> asumimos que es la hoja
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 0.3 * img_area:
        # Si el mayor contorno es pequeño, algo va mal: usamos toda la imagen
        return cv2.resize(img_bgr, A4_SIZE)

    # --- 4. Aproximamos a polígono y buscamos 4 puntos ---
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        # Si no son exactamente 4, usamos el rectángulo mínimo
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = np.array(box, dtype="float32")

    # --- 5. Ordenar puntos: TL, TR, BR, BL ---
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    # --- 6. Warp a A4_SIZE ---
    w_dest, h_dest = A4_SIZE
    dst = np.array([
        [0,         0],
        [w_dest-1,  0],
        [w_dest-1,  h_dest-1],
        [0,         h_dest-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w_dest, h_dest))

    return warped
