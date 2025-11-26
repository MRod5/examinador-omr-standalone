
"""
Definición de ROIs (Regiones de Interés) sobre la hoja enderezada test_case_A4.

Tarea A:
- Definir un ROI aproximado donde vive el QR (QR_AREA).
- Detectar el QR real dentro de ese ROI usando QRCodeDetector.
- Devolver/dibujar una caja ajustada alrededor del QR real.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class RectRel:
    """
    Rectángulo en coordenadas relativas a la imagen completa:
    x0, y0, x1, y1 en el rango [0, 1], con (0,0) en la esquina
    superior izquierda y (1,1) en la esquina inferior derecha.
    """
    x0: float
    y0: float
    x1: float
    y1: float


# === TAREA A: ROI aproximado del QR (macro ventana de búsqueda) =============

# Estos valores son un ROI amplio donde con seguridad cae el QR.
# La caja fina se ajusta luego automáticamente con el detector de QR.
QR_AREA = RectRel(
    x0=0.78,
    y0=0.03,
    x1=0.98,
    y1=0.20
)


def recortar_roi_rel(img, rect: RectRel):
    """
    Recorta un ROI definido en coordenadas relativas [0..1] sobre una imagen.

    Devuelve:
        subimg  : subimagen correspondiente al ROI.
        (x0, y0, x1, y1) : coordenadas de píxel usadas para el recorte.
    """
    h, w = img.shape[:2]

    x0 = int(rect.x0 * w)
    y0 = int(rect.y0 * h)
    x1 = int(rect.x1 * w)
    y1 = int(rect.y1 * h)

    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w,     x1))
    y1 = max(0, min(h,     y1))

    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)

    subimg = img[y0:y1, x0:x1]
    return subimg, (x0, y0, x1, y1)


def detectar_qr_preciso(img, macro_rect: RectRel = QR_AREA, margen: float = 1.15):
    """
    Busca el QR dentro de un ROI aproximado (macro_rect) y devuelve un
    RectRel ajustado alrededor del QR real.

    Parámetros
    ----------
    img : np.ndarray
        Imagen BGR de la hoja enderezada (test_case_A4).
    macro_rect : RectRel
        ROI relativo amplio donde esperamos encontrar el QR.
    margen : float
        Factor para ampliar ligeramente la caja alrededor del QR detectado.

    Devuelve
    --------
    rect_qr_rel : RectRel | None
        Rectángulo relativo ajustado al QR real, o None si no se detecta.
    data : str | None
        Contenido decodificado del QR (si lo hay).
    """
    h, w = img.shape[:2]

    roi, (rx0, ry0, rx1, ry1) = recortar_roi_rel(img, macro_rect)

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(roi)

    if points is None or len(points) == 0:
        # No se ha detectado QR dentro del ROI
        return None, None

    # points viene en coordenadas del ROI; lo pasamos a coords absolutas de la hoja
    pts = points.reshape(-1, 2)  # (4, 2)
    pts[:, 0] += rx0
    pts[:, 1] += ry0

    # Calculamos bounding box del QR detectado
    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()

    # Ampliamos un poco con el margen
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half_w = 0.5 * (x_max - x_min) * margen
    half_h = 0.5 * (y_max - y_min) * margen

    x0 = int(max(0, cx - half_w))
    x1 = int(min(w, cx + half_w))
    y0 = int(max(0, cy - half_h))
    y1 = int(min(h, cy + half_h))

    # Convertimos a relativas
    rect_qr_rel = RectRel(
        x0=x0 / w,
        y0=y0 / h,
        x1=x1 / w,
        y1=y1 / h,
    )

    return rect_qr_rel, data


def dibujar_qr_area(img):
    """
    Dibuja la caja verde del QR real detectado.
    Si no se detecta el QR, dibuja la macro-ventana QR_AREA.

    Parámetros
    ----------
    img : np.ndarray
        Imagen BGR sobre la que dibujar (típicamente test_case_A4).

    Devuelve
    --------
    img_out : np.ndarray
        Imagen con el rectángulo dibujado.
    """
    h, w = img.shape[:2]

    # Intentamos detección fina primero
    rect_preciso, data = detectar_qr_preciso(img)

    if rect_preciso is not None:
        rect = rect_preciso
    else:
        # Fallback: usamos el macro ROI aproximado
        rect = QR_AREA

    # Pasamos rect relativo a píxeles
    x0 = int(rect.x0 * w)
    y0 = int(rect.y0 * h)
    x1 = int(rect.x1 * w)
    y1 = int(rect.y1 * h)

    color = (0, 255, 0)  # verde
    thickness = 3
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

    # Si quieres ver qué se ha leído del QR:
    if data:
        print("QR detectado:", data)

    return img
