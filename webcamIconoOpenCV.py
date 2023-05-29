import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, background = video_capture.read()

    primer_plano = background[:, :, 0:3]
    altura_primer_plano, anchura_primer_plano, _ = primer_plano.shape

    cv2.imshow('MEZCLA', background)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(background,
                                                       cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100),
                                                       parameters=cv2.aruco.DetectorParameters())
    print(corners)
    print(ids)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()