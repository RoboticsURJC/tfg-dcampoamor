import cv2
import numpy as np
import threading
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from xmlrpc.server import SimpleXMLRPCServer
import socket
import os
import signal
import sys

# Parámetros de Cámara y proyección 
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933
DEGTORAD = 3.1415926535897932 / 180

# Variables para la navegación con OpenGL
angle_x = 0
angle_y = 0
zoom = -5
camera_x = 0
camera_y = 0
camera_z = 0
mouse_last_x = 0
mouse_last_y = 0
is_dragging = False

# Variables Globales para almacenar los puntos detectados
detected_points = []

# Modelo Pinhole de cámara
class PinholeCamera:
    def __init__(self):
        self.position = np.zeros(4)
        self.k = np.zeros((3, 4))
        self.rt = np.zeros((4, 4))

def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 65 * DEGTORAD
    thetaZ = 0 * DEGTORAD
    thetaX = 0 * DEGTORAD

    R_y = np.array([(np.cos(thetaY), 0, -np.sin(thetaY)), (0, 1, 0), (np.sin(thetaY), 0, np.cos(thetaY))])
    R_z = np.array([(np.cos(thetaZ), -np.sin(thetaZ), 0), (np.sin(thetaZ), np.cos(thetaZ), 0), (0, 0, 1)])
    R_x = np.array([(1, 0, 0), (0, np.cos(thetaX), np.sin(thetaX)), (0, -np.sin(thetaX), np.cos(thetaX))])

    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)

    T = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -265)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([(FX, 0, CX, 0), (0, FY, CY, 0), (0, 0, 1, 0)])

    myCamera.position = np.array([0, 0, -265, 1])
    myCamera.k = K
    myCamera.rt = RT

def pixel2optical(p2d):
    aux = p2d[0]
    p2d[0] = LARGO_IMAGEN - 1 - p2d[1]
    p2d[1] = aux
    return p2d

def backproject(p2d, camera):
    p3d = np.zeros(3)
    p2d = pixel2optical(p2d)
    p2d_h = np.array([p2d[0], p2d[1], 1])

    inv_K = np.linalg.inv(camera.k[:, :3])
    inv_RT = np.linalg.inv(camera.rt[:3, :3])

    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)

    p3d[:2] = p3d_h[:2] / p3d_h[2]
    p3d[2] = 0
    return p3d

def getIntersectionZ(p2d):
    p3d = backproject(p2d, myCamera)
    return p3d

def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    distancia = np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)
    return distancia

def detect_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids

def getPoints(frame):
    global detected_points

    lower_color = np.array([35, 100, 100])
    upper_color = np.array([85, 255, 255])

    centroids = detect_color(frame, lower_color, upper_color)

    detected_points = []
    for idx, centroid in enumerate(centroids):
        centroid_x = int(centroid[0])
        centroid_y = int(centroid[1])

        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Ajuste: usa tuple (x, y)
        cv2.putText(frame, f"P{idx+1} ({centroid_x},{centroid_y})", (centroid_x - 20, centroid_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        pixel = np.array([centroid_x, centroid_y])
        pixelOnGround3D = getIntersectionZ(pixel)

        x_punto = pixelOnGround3D[0]
        y_punto = pixelOnGround3D[1]
        z_punto = pixelOnGround3D[2]

        detected_points.append((x_punto, y_punto, z_punto))

        x_cam = myCamera.position[0]
        y_cam = myCamera.position[1]
        z_cam = myCamera.position[2]

        distancia = calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto)

        print(f"Punto P{idx+1} - Coordenadas 3D: X={x_punto}, Y={y_punto}, Z={z_punto}")
        print(f"Punto P{idx+1} - Distancia al punto: {distancia:.2f} milímetros")

def camera_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    loadCamera()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede capturar el frame")
            break

        flipped_frame = cv2.flip(frame, 1)
        getPoints(flipped_frame)

        cv2.imshow("Imagen de la camara", flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_xmlrpc_server():
    done_pose = [0, 0, 0, 0, 0, 0]

    def get_next_pose():
        global detected_points
        if detected_points:
            pose_value = detected_points.pop(0)  # Esto debe ser un tuple (x, y, z)
            
            # Verificamos que pose_value tenga 3 elementos
            if len(pose_value) == 3:  # Asegúrate de que sea de 3 elementos
                x = float(pose_value[0])  # x
                y = float(pose_value[1])  # y
                z = 0
                Rx = 0
                Ry = 0
                Rz = 0

                # Asignamos los valores a done_pose
                done_pose[0] = x
                done_pose[1] = y
                done_pose[2] = z
                done_pose[3] = Rx
                done_pose[4] = Ry
                done_pose[5] = Rz

                print(f"Enviando coordenadas: {done_pose}")
                return done_pose
            else:
                print("Error: 'pose_value' no tiene 3 elementos.")
        else:
            print("Advertencia: No hay puntos detectados en 'detected_points'.")

        return done_pose

    with SimpleXMLRPCServer(("192.168.23.107", 50000), allow_none=True) as server:
        server.register_function(get_next_pose, "get_next_pose")
        print("Servidor XML-RPC corriendo en http://192.168.23.107:50000...")

        # Manejando señales para detener el servidor de manera ordenada
        def signal_handler(sig, frame):
            print("\nServidor detenido de manera ordenada.")
            server.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        server.serve_forever()

if __name__ == "__main__":
    # Arrancar en diferentes hilos
    camera_thread = threading.Thread(target=camera_capture)
    server_thread = threading.Thread(target=start_xmlrpc_server)

    camera_thread.start()
    server_thread.start()

    camera_thread.join()
    server_thread.join()

