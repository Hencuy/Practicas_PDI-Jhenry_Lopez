import cv2
import mediapipe as mp

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y encontrar la malla facial
    results = face_mesh.process(image)

    # Convertir la imagen de vuelta a BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar la malla facial
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Mostrar la imagen con la malla facial
    cv2.imshow('MediaPipe Face Mesh', image)

    # Salir del loop al presionar la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
