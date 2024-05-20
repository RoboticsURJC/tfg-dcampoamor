#Este programa en python modifica las instrucciones del Hello World en OpenGL con tal de experimentar en esta librería

from Window import *

class IntroduccionOpenGL(Window):
    '''Este es un programa simple e introductorio a OpenGL.'''

    def __init__(self):
        super(IntroduccionOpenGL, self).__init__("hello.c", "Introduccion a OpenGL", 500, 500, 1)

    def display(self):
        '''Se renderiza el esquema de un cubo y un triángulo.'''
        glClear(GL_COLOR_BUFFER_BIT)  # Borra todos los píxeles.

        # Dibuja un polígono blanco (rectángulo) con las esquinas en (0.25, 0.25, 0) y (0.75, 0.75, 0)
        glColor3f(1, 1, 1)
        glBegin(GL_POLYGON)
        glVertex3f(0.5, 0.5, 0)
        glVertex3f(0.75, 0.5, 0)
        glVertex3f(0.75, 0.75, 0)
        glVertex3f(0.5, 0.75, 0)
        glEnd()

        # Dibuja un triángulo rojo con vértices en (0.0, 0.0, 0.0), (1.0, 0.0, 0.0) y (0.5, 1.0, 0.0)
        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.5, 0.0, 0.0)
        glVertex3f(0.25, 0.5, 0.0)
        glEnd()

       
        glFlush()

if __name__ == '__main__':
    IntroduccionOpenGL().run()



