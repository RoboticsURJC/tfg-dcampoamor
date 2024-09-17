import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import threading

# Variables globales
points_to_draw = []
frame_to_show = None
cap = None

# Variables de cámara
camera_x, camera_y, zoom = 0.0, 0.0, -3.0
angle_x, angle_y = 0.0, 0.0
last_mouse_x, last_mouse_y = 0, 0

# Función para detectar puntos en un frame usando OpenCV
def detect_points(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define el rango del color amarillo en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    points = np.column_stack(np.where(mask > 0))
    return points

# Función para actualizar la cámara y detectar puntos
def update_camera():
    global points_to_draw, frame_to_show
    ret, frame = cap.read()
    if ret:
        points = detect_points(frame)
        points_to_draw = points if points is not None else []
        frame_to_show = frame

# Función de renderizado con OpenGL
def draw_points():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Aplicar transformación de la cámara
    glTranslatef(camera_x, camera_y, zoom)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)

    # Dibujar puntos detectados
    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)  # Color blanco
    for point in points_to_draw:
        y, x = point  # Invertir el orden de x e y para corregir la rotación
        glVertex2f(x / 640.0 - 1.0, 1.0 - y / 480.0)
    glEnd()
    glutSwapBuffers()

# Función de temporizador para actualizar y redibujar
def timer(value):
    update_camera()
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)

# Función para manejar teclas especiales
def special_keys(key, x, y):
    global camera_x, camera_y
    if key == GLUT_KEY_UP:
        camera_y += 0.1
    elif key == GLUT_KEY_DOWN:
        camera_y -= 0.1
    elif key == GLUT_KEY_LEFT:
        camera_x -= 0.1
    elif key == GLUT_KEY_RIGHT:
        camera_x += 0.1
    glutPostRedisplay()

# Función para manejar el movimiento del ratón
def motion(x, y):
    global angle_x, angle_y, last_mouse_x, last_mouse_y
    dx = x - last_mouse_x
    dy = y - last_mouse_y
    angle_x += dy * 0.1
    angle_y += dx * 0.1
    last_mouse_x = x
    last_mouse_y = y
    glutPostRedisplay()

# Función para manejar el botón del ratón
def mouse(button, state, x, y):
    global last_mouse_x, last_mouse_y, zoom
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        last_mouse_x, last_mouse_y = x, y
    elif button == 3:  # Scroll up
        zoom += 0.5
    elif button == 4:  # Scroll down
        zoom -= 0.5
    glutPostRedisplay()

# Función para mostrar el feed de la cámara en una ventana separada
def show_camera_feed():
    while True:
        if frame_to_show is not None:
            cv2.imshow('Camera Feed', frame_to_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Función principal
def main():
    global cap
    cap = cv2.VideoCapture(0)

    # Crear un hilo para mostrar el feed de la cámara
    camera_thread = threading.Thread(target=show_camera_feed)
    camera_thread.start()

    # Configurar OpenGL y GLUT
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(700, 100)  # Posición para evitar superposición con la ventana de la cámara
    glutCreateWindow("OpenGL Points")
    glutDisplayFunc(draw_points)
    glutSpecialFunc(special_keys)
    glutMotionFunc(motion)
    glutMouseFunc(mouse)
    glutTimerFunc(16, timer, 0)
    glutMainLoop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

