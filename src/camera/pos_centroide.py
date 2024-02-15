import cv2
import numpy as np

# Función para convertir de píxeles a unidades ópticas
def pixel2optical(pixel_x, pixel_y):
    # Valores de ejemplo, se deben ajustar según las especificaciones de la cámara
    sensor_width = 6  # Ancho del sensor en mm (ejemplo para un sensor full-frame)

    # Convertir píxeles a unidades ópticas
    optic_x = (pixel_x - 316.068) * (sensor_width / 640)  # Suponiendo una resolución de 640x480
    optic_y = (pixel_y - 236.933) * (sensor_width / 480)  # y que la cámara está centrada

    return optic_x, optic_y

# Función para proyectar coordenadas 2D a 3D
def project(optic_x, optic_y):
    K = np.array([[816.218, 0, 316.068],
                  [0, 814.443, 236.933],
                  [0, 0, 1]])
    R = np.array([[0.875, 0, -0.485],
                  [0, 1 , 0],
                  [0.485, 0, 0.875]])
    T = np.array([[0], [0], [330]])

    # Asegurar que R tiene la forma correcta (3, 3)
    R = R.reshape(3, 3)

    # Asegurar que T tiene la forma correcta (3, 1)
    T = T.reshape(3, 1)

    # Concatenar R y -RT
    RT = np.concatenate((R.T, -np.dot(R.T, T)), axis=1)

    # Añadir una fila a coords_homogeneous
    coords_homogeneous = np.array([optic_x, optic_y, 1, 1]).reshape(-1, 1)

    # Proyectar coordenadas 2D a 3D
    P = np.dot(K, np.dot(RT, coords_homogeneous))
    return P

def detect_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            return centroid_x, centroid_y
    return None, None

def main():
    # Definir el rango de color HSV del objeto que deseas detectar
    lower_color = np.array([30, 100, 100])
    upper_color = np.array([70, 255, 255])

    # Iniciar la captura de la cámara web
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        centroid_x, centroid_y = detect_color(frame, lower_color, upper_color)
        
        if centroid_x is not None and centroid_y is not None:
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Centroide: ({centroid_x}, {centroid_y})", (centroid_x - 100, centroid_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convertir coordenadas de píxeles a ópticas
            optic_x, optic_y = pixel2optical(centroid_x, centroid_y)
            print(f"Coordenadas ópticas: X={optic_x}, Y={optic_y}")
            
            # Proyectar coordenadas 2D a 3D
            focal_length = 100  # Longitud focal en mm (debe ser ajustada según las especificaciones de la cámara)
            x, y, z = project(optic_x, optic_y)
            print(f"Coordenadas 3D: X={x}, Y={y}, Z={z}")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

