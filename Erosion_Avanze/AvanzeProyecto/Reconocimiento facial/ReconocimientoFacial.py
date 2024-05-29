import cv2
import os
import bcrypt
import mediapipe as mp

dataPath = 'C:\\Users\\antou\\OneDrive\\Documentos\\2024\\Fotos'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('Video.mp4') 

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame para una mejor visualización
    frame = cv2.resize(frame, (640, 480))
    auxFrame = frame.copy()

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar la malla facial
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Obtener las coordenadas de la caja delimitadora del rostro
            h, w, _ = frame.shape
            face_coords = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]
            x_min = min([coord[0] for coord in face_coords])
            y_min = min([coord[1] for coord in face_coords])
            x_max = max([coord[0] for coord in face_coords])
            y_max = max([coord[1] for coord in face_coords])

            # Extraer el rostro para reconocimiento
            rostro = auxFrame[y_min:y_max, x_min:x_max]
            if rostro.size != 0:
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                gray_rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                result = face_recognizer.predict(gray_rostro)

                cv2.putText(frame, '{}'.format(result), (x_min, y_min - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

                if result[1] < 70:
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x_min, y_min - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Desconocido', (x_min, y_min - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    cv2.imshow('MediaPipe Face Mesh', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
