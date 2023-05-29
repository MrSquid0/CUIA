import cv2
import numpy as np
lena = cv2.imread('lena.tif')
opencv = cv2.imread('opencv.png', cv2.IMREAD_UNCHANGED)

fondo = lena
hfondo, wfondo, _ = fondo.shape
primerplano = opencv[:, :, 0:3]
hprimerplano, wprimerplano, _ = primerplano.shape

x = wfondo//2 - wprimerplano//2
y = hfondo//2 - hprimerplano//2

mezcla = fondo
mezcla[y:y+hprimerplano, x:x+wprimerplano] = primerplano

cv2.imshow('MEZCLA', mezcla)
cv2.waitKey()
cv2.destroyAllWindows()