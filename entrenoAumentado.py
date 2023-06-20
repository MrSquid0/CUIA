import cv2
import speech_recognition as sr
from deepface import DeepFace
import os
import shutil
import uuid

print("Inicializando aplicación... Por favor, espera")

video_capture = cv2.VideoCapture(0)

rec = sr.Recognizer()
mic = sr.Microphone()

known_persons_directory = 'known_persons'
preferences_dir = 'preferences'

user = ""
dificultad = ""

# reconocimiento facial
while True:
    print("Intentando verificar cara, mira fijamente a la cámara y no te muevas...")
    ret, frame = video_capture.read()
    # Show the captured frame
    cv2.imshow('Input', frame)
    cv2.imwrite('capture.jpg', frame)
    caras = DeepFace.extract_faces(img_path="capture.jpg", detector_backend='retinaface', enforce_detection=False)

    if len(caras) > 0:
        print(f"Cara detectada! '{len(caras)}', verificando si es nueva o ya fue reconocida antes...")

        if not os.path.exists(known_persons_directory):
            os.makedirs(known_persons_directory)
        if not os.path.exists(preferences_dir):
            os.makedirs(preferences_dir)

        known = False
        for filename in os.listdir(known_persons_directory):
            result = DeepFace.verify(img1_path=f"{known_persons_directory}/{filename}", img2_path="capture.jpg", enforce_detection=False)
            #print(f'res {result}')
            if result['verified']:
                known = True
                user = filename.replace(".png", "")
                break

        if known == False:
            user = uuid.uuid4()
            shutil.copyfile("capture.jpg", f"{known_persons_directory}/{user}.png")
            break
        else:
            print(f"Has sido reconocido como el usuario {user}")
            break
    else:
        print("Ninguna cara detectada. Reintentando....")

text = None
if os.path.exists(f"{preferences_dir}/{user}"):
    with open(f"{preferences_dir}/{user}", 'r') as file:
        text = file.read()
while (text != "fácil" and text != "difícil"):
    try:
        with mic as source:
            print("Grabando audio... Di 'fácil' o 'difícil para mostrar el ejercicio adecuado'")
            rec.adjust_for_ambient_noise(source, duration=0.5)
            audio = rec.listen(source)

            text = rec.recognize_google(audio, language='es-ES')

            print(f"He entendido: '{text}'")
            with open(f"{preferences_dir}/{user}", 'w') as file:
                # Write the string to the file
                file.write(text)
            print(f"'{text}' guardado en 'preferences' para el usuario '{user}'")
    except Exception as e:
        print(f"Error con el input. Reintentando... ({e})")

video_capture.release()
cv2.destroyAllWindows()

video_capture = cv2.VideoCapture(0)

def get_media_by_difficulty_and_step(difficulty, ejercicio):
    if ejercicio is None:
        return 'medias/flexiones-facil.png'
    return f'medias/{ejercicio}-{difficulty.replace("í", "i").replace("á", "a")}.png'

while True:
    ret, background = video_capture.read()

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100),
                                       cv2.aruco.DetectorParameters())

    (corners, ids, rejected) = detector.detectMarkers(background)


    logo = cv2.imread(get_media_by_difficulty_and_step(text, None), cv2.IMREAD_UNCHANGED)

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
        esquina_inferior = primer_plano[primer_parametro:segundo_parametro,
                           tercer_parametro:cuarto_parametro] * inverso_alfa + minilog * alfa
        primer_plano[primer_parametro:segundo_parametro, tercer_parametro:cuarto_parametro] = esquina_inferior

    cv2.imshow('Entreno Aumentado', background)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
