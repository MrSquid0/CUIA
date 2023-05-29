import cv2
import numpy as np
from matplotlib import pyplot as plt



def eventoraton(evento, x, y, flags, params):
    if evento == cv2.EVENT_LBUTTONUP:
        print("H: ", framehsv[y, x, 0])
        print("S: ", framehsv[y, x, 1])
        print("V: ", framehsv[y, x, 2])
        print("-----")

cv2.namedWindow("WEBCAM")
cv2.setMouseCallback("WEBCAM", eventoraton)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

histograma, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.ion()
plt.show()
while cap.isOpened():
    ret, framebgr = cap.read()

    if not ret:
        print("No he podido leer el frame")
        break

    # Procesado de imágenes aquí
    framehsv = cv2.cvtColor(framebgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(framehsv)

    histoh = cv2.calcHist( [framehsv], [0], None, [180], [0, 180])
    histos = cv2.calcHist( [framehsv], [1], None, [256], [0, 256])
    histov = cv2.calcHist( [framehsv], [2], None, [256], [0, 256])


    cv2.imshow('WEBCAM', framebgr)

    ax1.clear()
    ax1.set_title("TONO")
    ax1.get_yaxis().set_visible(False)
    ax1.plot(histoh)
    ax2.clear()
    ax2.plot(histos)
    ax2.set_title('SATURACION')
    ax1.get_yaxis().set_visible(False)
    ax3.clear()
    ax3.plot(histov)
    ax3.set_title('VALOR')
    ax1.get_yaxis().set_visible(False)
    if cv2.waitKey(1) == ord(' '):
        break
    histograma.canvas.flush_events()

cap.release()
cv2.destroyWindow('WEBCAM')