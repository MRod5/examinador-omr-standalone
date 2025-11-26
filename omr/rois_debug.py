"""
Funciones de depuración para mostrar las ROIs detectadas en la hoja de respuestas.
"""

import cv2
from .geometry import QR_AREA, BUBBLES_AREA, LEFT_COL, RIGHT_COL, recortar_roi_rel

OPCIONES = ["A", "B", "C", "D"]

def dibujar_rois_y_malla(hoja_bgr):
    """
    Dibuja:
      - Rectángulo del QR
      - Rectángulo general de la tabla de burbujas
      - Sub-rectángulos de las dos columnas
      - Centros aproximados de las burbujas (60 x 4)
    y muestra la imagen.
    """
    img = hoja_bgr.copy()
    h, w = img.shape[:2]

    # --- Dibujar rectángulo QR ---
    qr_roi, (qx, qy, qw, qh) = recortar_roi_rel(img, QR_AREA)
    cv2.rectangle(img, (qx, qy), (qx + qw, qy + qh), (0, 255, 0), 2)

    # --- Dibujar rectángulo global de burbujas ---
    bubbles_roi, (bx, by, bw, bh) = recortar_roi_rel(img, BUBBLES_AREA)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    # --- Dibujar sub-rectángulos de columnas dentro de BUBBLES_AREA ---
    # Trabajamos en coordenadas relativas a la zona de burbujas
    # pero los dibujamos en la imagen global
    # Columna izquierda (preguntas 1-30)
    left_x0 = bx + int(LEFT_COL.x0 * bw)
    left_y0 = by + int(LEFT_COL.y0 * bh)
    left_x1 = bx + int(LEFT_COL.x1 * bw)
    left_y1 = by + int(LEFT_COL.y1 * bh)
    cv2.rectangle(img, (left_x0, left_y0), (left_x1, left_y1), (0, 0, 255), 2)

    # Columna derecha (preguntas 31-60)
    right_x0 = bx + int(RIGHT_COL.x0 * bw)
    right_y0 = by + int(RIGHT_COL.y0 * bh)
    right_x1 = bx + int(RIGHT_COL.x1 * bw)
    right_y1 = by + int(RIGHT_COL.y1 * bh)
    cv2.rectangle(img, (right_x0, right_y0), (right_x1, right_y1), (0, 0, 255), 2)

    # --- Dibujar malla aproximada de centros de burbujas ---

    # Columna izquierda
    ch_left = left_y1 - left_y0
    cw_left = left_x1 - left_x0
    filas = 30
    cols = len(OPCIONES)
    row_step = ch_left / filas
    col_step = cw_left / cols

    for i in range(filas):
        for j in range(cols):
            cx = int(left_x0 + (j + 0.5) * col_step)
            cy = int(left_y0 + (i + 0.5) * row_step)
            cv2.circle(img, (cx, cy), 6, (0, 255, 255), 1)

    # Columna derecha
    ch_right = right_y1 - right_y0
    cw_right = right_x1 - right_x0
    row_step_r = ch_right / filas
    col_step_r = cw_right / cols

    for i in range(filas):
        for j in range(cols):
            cx = int(right_x0 + (j + 0.5) * col_step_r)
            cy = int(right_y0 + (i + 0.5) * row_step_r)
            cv2.circle(img, (cx, cy), 6, (255, 255, 0), 1)

    cv2.imshow("ROIs y malla de burbujas", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
