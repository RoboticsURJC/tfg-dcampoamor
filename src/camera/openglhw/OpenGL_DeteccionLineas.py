import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Variables globales
lines_to_draw = []
frame_to_show = None
cap = None

# Función para detectar líneas en un frame usando OpenCV
def detect_lines(frame):
    edges = cv2.Canny(frame, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    return lines

# Función para actualizar la cámara y detectar líneas
def update_camera():
    global lines_to_draw, frame_to_show
    ret, frame = cap.read()
    if ret:
        lines = detect_lines(frame)
        lines_to_draw = lines if lines is not None else []
        frame_to_show = frame

# Función de renderizado con OpenGL
def draw_lines():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 1.0,   0.0, 0.0, 0.0,   0.0, 1.0, 0.0)

    # Dibujar líneas detectadas
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 0.0)  # Color amarillo
    for line in lines_to_draw:
        x1, y1, x2, y2 = line[0]
        glVertex2f(x1 / 640.0 - 1.0, 1.0 - y1 / 480.0)
        glVertex2f(x2 / 640.0 - 1.0, 1.0 - y2 / 480.0)
    glEnd()
    glutSwapBuffers()

# Función de temporizador para actualizar y redibujar
def timer(value):
    update_camera()
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)

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
    cap = cv2.VideoCapture(2)

    # Crear un hilo para mostrar el feed de la cámara
    import threading
    camera_thread = threading.Thread(target=show_camera_feed)
    camera_thread.start()

    # Configurar OpenGL y GLUT
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(700, 100)  # Posición para evitar superposición con la ventana de la cámara
    glutCreateWindow("OpenGL Lines")
    glutDisplayFunc(draw_lines)
    glutTimerFunc(16, timer, 0)
    glutMainLoop()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

