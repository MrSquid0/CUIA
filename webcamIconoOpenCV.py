import cv2
import speech_recognition as sr

video_capture = cv2.VideoCapture(0)

rec = sr.Recognizer()
mic = sr.Microphone()

text = None
while(text != "fácil" and text != "dificil"):
    try:
        with mic as source:
            print(f"Recording audio...")
            rec.adjust_for_ambient_noise(source, duration=0.5)
            audio = rec.listen(source)

            text = rec.recognize_google(audio, language='es-ES')

            print(f"What I understood: '{text}'")
    except Exception as e:
        print(f"Error reading input. Gonna retry... ({e})")

def get_media_by_difficulty_and_step(difficulty, step):
    return f'medias/{difficulty}.png'


logo = cv2.imread(get_media_by_difficulty_and_step(text, None), cv2.IMREAD_UNCHANGED)

while True:
    ret, background = video_capture.read()

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100), cv2.aruco.DetectorParameters())

    (corners, ids, rejected) = detector.detectMarkers(background)

    marca_de_agua = cv2.resize(logo, None, fx=0.34, fy=0.3)

    primer_plano = background[:, :, 0:3]
    altura_primer_plano, anchura_primer_plano, _ = primer_plano.shape

    minilog = marca_de_agua[:, :, 0:3]
    altura_logo, anchura_logo, _ = minilog.shape
    alfa = marca_de_agua[:, :, 3]
    inverso_alfa = 255 - alfa  # inverso de alfa
    alfa = cv2.cvtColor(alfa, cv2.COLOR_GRAY2BGR) / 255
    inverso_alfa = cv2.cvtColor(inverso_alfa, cv2.COLOR_GRAY2BGR) / 255

    if len(corners) > 0:
        primer_parametro = int(corners[0][0][2][1]) - altura_logo
        segundo_parametro = int(corners[0][0][2][1])
        tercer_parametro = int(corners[0][0][1][0]) - anchura_logo
        cuarto_parametro = int(corners[0][0][1][0])

        marca_de_agua = cv2.resize(logo, (cuarto_parametro - tercer_parametro, segundo_parametro - primer_parametro))

    if len(corners) > 0 \
            and len(primer_plano[primer_parametro:segundo_parametro, tercer_parametro:cuarto_parametro]) > 0 \
            and len(primer_plano[primer_parametro:segundo_parametro, tercer_parametro:cuarto_parametro][1]) > 0:
        esquina_inferior = primer_plano[primer_parametro:segundo_parametro, tercer_parametro:cuarto_parametro] * inverso_alfa + minilog * alfa
        primer_plano[primer_parametro:segundo_parametro, tercer_parametro:cuarto_parametro] = esquina_inferior

    cv2.imshow('MEZCLA', background)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()