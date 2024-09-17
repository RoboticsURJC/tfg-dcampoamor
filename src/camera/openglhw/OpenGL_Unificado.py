import random
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def noop():
    '''Una función que no hace nada.'''
    pass

class Window(object):
    '''Una ventana GLUT abstracta.'''
    def __init__(self, source=None, title="Untitled Window", width=500, height=500, ortho=None):
        '''Construye una ventana con el título y dimensiones dados. La fuente es el archivo redbook original'''
        self.source = source
        self.ortho  = ortho
        self.width  = width
        self.height = height
        self.keybindings = {chr(27):exit}
        glutInit()
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(title)
        # Just request them all and don't worry about it.
        glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH)
        glClearColor(0, 0, 0, 0)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutDisplayFunc(self.display)
        glutMouseFunc(self.mouse)
        glShadeModel(GL_FLAT)

    def keyboard(self, key, mouseX, mouseY):
        '''Llama al código asignado a la tecla pulsada.'''
        self.keybindings.get(key, noop)()
        glutPostRedisplay()
    
    def mouse(self, button, state, x, y):
        '''Maneja los clics del ratón.'''
        if button == GLUT_LEFT_BUTTON:
            self.mouseLeftClick(x, y)
        elif button == GLUT_MIDDLE_BUTTON:
            self.mouseMiddleClick(x, y)
        elif button == GLUT_RIGHT_BUTTON:
            self.mouseRightClick(x, y)
        else:
            raise ValueError(button)
        glutPostRedisplay()

    def mouseLeftClick(self, x, y):
        pass

    def mouseMiddleClick(self, x, y):
        pass

    def mouseRightClick(self, x, y):
        pass

    def reshape(self, width, height):
        '''Recalcula la ventana de recorte, la ventana GLUT se redimensiona.'''
        self.width  = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def display(self):
        raise NotImplementedError

    @staticmethod
    def run():
        '''Inicia el bucle principal.'''
        glutMainLoop()

class IntroduccionOpenGL(Window):
    '''Este es un programa simple e introductorio a OpenGL.'''

    def __init__(self):
        super(IntroduccionOpenGL, self).__init__("hello.c", "Introduccion a OpenGL", 500, 500, 1)
        self.puntos = self.generar_puntos_aleatorios()

    def generar_puntos_aleatorios(self):
        '''Genera un número aleatorio de puntos dentro de un rango 640x480.'''
        num_puntos = random.randint(3, 10)  # Al menos un triángulo.
        puntos = [(random.uniform(0, 640), random.uniform(0, 480)) for _ in range(num_puntos)]
        return puntos

    def display(self):
        '''Se renderiza un polígono con puntos aleatorios.'''
        glClear(GL_COLOR_BUFFER_BIT)  # Borra todos los píxeles.

        # Configura el color del polígono.
        glColor3f(1, 1, 1)
        glPushMatrix()
        # Escala los puntos de 640x480 al tamaño de la ventana actual.
        glScalef(self.width / 640.0, self.height / 480.0, 1.0)
        
        glBegin(GL_POLYGON)
        for x, y in self.puntos:
            glVertex2f(x, y)
        glEnd()
        
        glPopMatrix()
        glFlush()
        glutSwapBuffers()

if __name__ == '__main__':
    IntroduccionOpenGL().run()

