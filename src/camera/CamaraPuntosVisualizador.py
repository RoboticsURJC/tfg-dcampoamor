import cv2
import numpy as np

# Función para dibujar los puntos en la imagen
def draw_points(img, points):
    for point in points:
        x, y = point
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Dibuja un círculo verde en el punto
        cv2.putText(img, f'({x},{y})', (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Muestra las coordenadas del punto

# Puntos a dibujar
points = [[256, 256], [300, 300], [150, 200], [50, 450]]

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Captura un frame del video
    
    if not ret:
        break

    # Cambia la resolución del frame a 640x480
    frame = cv2.resize(frame, (640, 480))

    # Dibuja los puntos en el frame
    draw_points(frame, points)

    # Muestra el frame con los puntos
    cv2.imshow('Camara', frame)

    # Espera a que se presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()

