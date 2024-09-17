from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Variables para controlar la cámara
angle_x = 0
angle_y = 0
zoom = -5
camera_x = 0
camera_y = 0
camera_z = 0
mouse_last_x = 0
mouse_last_y = 0
is_dragging = False

def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Transformaciones de la cámara
    glTranslatef(camera_x, camera_y, zoom)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    
    draw_cube()
    
    glutSwapBuffers()

def draw_cube():
    glBegin(GL_QUADS)

    # Front face (z = 1.0)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    
    # Back face (z = -1.0)
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)
    
    # Top face (y = 1.0)
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    
    # Bottom face (y = -1.0)
    glColor3f(1.0, 1.0, 0.0)  # Yellow
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    
    # Right face (x = 1.0)
    glColor3f(1.0, 0.0, 1.0)  # Magenta
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    
    # Left face (x = -1.0)
    glColor3f(0.0, 1.0, 1.0)  # Cyan
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)

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

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Interaccion con cubo 3D")

    init()
    
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutSpecialFunc(special_key)
    
    glutMainLoop()

if __name__ == "__main__":
    main()

