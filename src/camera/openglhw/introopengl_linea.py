from Window import *

class IntroduccionOpenGL(Window):
    '''Este es un programa simple e introductorio a OpenGL.'''
    
    def __init__(self):
        super(IntroduccionOpenGL, self).__init__("hello.c", "Dibujar linea recta con OpenGL", 500, 500, 1)

    def init_gl(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Definir los puntos para la línea
        point1 = (0, 0)
        point2 = (0.9, 0.9)

        # Dibujar la línea entre los puntos
        glColor3f(1, 1, 1)
        glBegin(GL_LINES)
        glVertex2f(*point1)
        glVertex2f(*point2)
        glEnd()

        glFlush()

    def run(self):
        glutMainLoop()

if __name__ == '__main__':
    IntroduccionOpenGL().run()

