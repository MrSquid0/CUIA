import cv2
import numpy as np

img = cv2.imread('lena.tif')
b, g, r = cv2.split(img)
zeros_ch = np.zeros(img.shape[0:2], dtype="uint8")

# Imagen azul
blue_img = cv2.merge([b, zeros_ch, zeros_ch])
cv2.imshow("Imagen en azul", blue_img)

# Imagen verde
green_img = cv2.merge([zeros_ch, g, zeros_ch])
cv2.imshow("Imagen en verde", green_img)

# Imagen roja
red_img = cv2.merge([zeros_ch, zeros_ch, r])
cv2.imshow("Imagen en rojo", red_img)

cv2.waitKey(0)
cv2.destroyAllWindows()