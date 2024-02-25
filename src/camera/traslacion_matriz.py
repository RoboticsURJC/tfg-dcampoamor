import cv2
import numpy as np
from pathlib import Path

# Definir las dimensiones de la matriz
filas = 480
columnas = 640

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    if ret:
        # Convertir el frame a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular el desplazamiento necesario para trasladar el origen al centro
        desplazamiento_filas = filas // 2
        desplazamiento_columnas = columnas // 2

        # Coordenadas del origen de la matriz original
        origen_original = (0, 0)
        # Coordenadas de un punto cualquiera en la matriz original
        punto_original = (100, 100)

        # Coordenadas del origen de la matriz trasladada
        origen_trasladado = (desplazamiento_columnas, desplazamiento_filas)
        # Calcular las coordenadas del mismo punto en la matriz trasladada
        punto_trasladado = (punto_original[0] + desplazamiento_columnas, punto_original[1] + desplazamiento_filas)

        # Dibujar el origen de la matriz original en rojo
        cv2.circle(frame, origen_original, 5, (0, 0, 255), -1)
        # Dibujar el punto original en verde
        cv2.circle(frame, punto_original, 5, (0, 255, 0), -1)

        # Dibujar el origen de la matriz trasladada en azul
        cv2.circle(frame, origen_trasladado, 5, (255, 0, 0), -1)
        # Dibujar el punto trasladado en amarillo
        cv2.circle(frame, punto_trasladado, 5, (0, 255, 255), -1)

        # Mostrar el frame resultante
        cv2.imshow('Matriz con puntos', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()

