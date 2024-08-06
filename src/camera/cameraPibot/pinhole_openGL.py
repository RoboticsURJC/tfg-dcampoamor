###############################################################################
# (C) 2019 - Julio Vega
###############################################################################
# Algoritmo de navegación de PiBot basado en sonar visual percibido por PiCam.#
# En este programa, los motores están parados, para así poder hacer pruebas.  #
###############################################################################


# Import necessary libraries
import cv2
import numpy as np
import threading
import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Camera and projection parameters
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933
DEGTORAD = 3.1415926535897932 / 180

# Variables for OpenGL navigation
angle_x = 0
angle_y = 0
zoom = -5
camera_x = 0
camera_y = 0
camera_z = 0
mouse_last_x = 0
mouse_last_y = 0
is_dragging = False

# Global variable to store detected points
detected_points = []

# Pinhole camera model
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

    # Apply morphology operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        # Get the centroid of the contour
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

    detected_points = []  # Clear previous points
    for idx, centroid in enumerate(centroids):
        centroid_x = int(centroid[0])
        centroid_y = int(centroid[1])

        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
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
    cv2.namedWindow("Image Feed")
    cv2.moveWindow("Image Feed", 159, -25)

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        exit()

    loadCamera()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede capturar el frame")
            break

        flipped_frame = cv2.flip(frame, 1)
        getPoints(flipped_frame)

        cv2.imshow("Image Feed", flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Camera transformations
    glTranslatef(camera_x, camera_y, zoom)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)

    draw_points()

    glutSwapBuffers()

def draw_points():
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    for point in detected_points:
        glVertex3f(point[0], point[1], point[2])
    glEnd()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def mouse(button, state, x, y):
    global zoom, is_dragging, mouse_last_x, mouse_last_y

    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        is_dragging = True
        mouse_last_x = x
        mouse_last_y = y
    elif button == GLUT_LEFT_BUTTON and state == GLUT_UP:
        is_dragging = False
    elif button == 3:  # Scroll up
        zoom += 0.5
    elif button == 4:  # Scroll down
        zoom -= 0.5

    glutPostRedisplay()

def motion(x, y):
    global angle_x, angle_y, mouse_last_x, mouse_last_y

    if is_dragging:
        dx = x - mouse_last_x
        dy = y - mouse_last_y

        angle_x += dy * 0.5
        angle_y += dx * 0.5

        mouse_last_x = x
        mouse_last_y = y

        glutPostRedisplay()

def special_key(key, x, y):
    global camera_x, camera_y

    step = 0.1

    if key == GLUT_KEY_UP:
        camera_y += step
    elif key == GLUT_KEY_DOWN:
        camera_y -= step
    elif key == GLUT_KEY_LEFT:
        camera_x -= step
    elif key == GLUT_KEY_RIGHT:
        camera_x += step

    glutPostRedisplay()

def opengl_main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Interaccion con puntos 3D")

    init()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutSpecialFunc(special_key)

    glutMainLoop()

if __name__ == "__main__":
    # Run camera capture in a separate thread
    camera_thread = threading.Thread(target=camera_capture)
    camera_thread.start()

    # Run OpenGL main loop
    opengl_main()

