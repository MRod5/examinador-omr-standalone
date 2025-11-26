# omr/debug_corners.py
import cv2

def inspeccionar_corners(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Aumentar contraste local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray2 = clahe.apply(gray)

    # 2) Threshold adaptativo fuerte para rescatar líneas finas
    thr = cv2.adaptiveThreshold(
        gray2, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,    # tamaño del bloque
        8      # constante substractiva
    )

    # 3) Morfología para engordar las líneas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thr2 = cv2.dilate(thr, kernel, iterations=1)

    contours, _ = cv2.findContours(thr2,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    debug = cv2.cvtColor(thr2, cv2.COLOR_GRAY2BGR)

    print(f"Contornos totales: {len(contours)}")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 5000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(debug, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(debug, str(int(area)), (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    cv2.imshow("Original", img_bgr)
    cv2.imshow("Threshold fuerte", thr2)
    cv2.imshow("Contornos bounding boxes", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
