from Window import *

class IntroduccionOpenGL(Window):
    '''Este es un programa simple e introductorio a OpenGL.'''

    def __init__(self):
        super(IntroduccionOpenGL, self).__init__("hello.c", "Introduccion a OpenGL", 650, 650, 1)

    def display(self):
        '''Se renderiza el esquema de un cubo y un triángulo.'''
        glClear(GL_COLOR_BUFFER_BIT)  # Borra todos los píxeles.

        # Dibuja un polígono blanco (rectángulo) con las esquinas en (0.25, 0.25, 0) y (0.75, 0.75, 0)
        glColor3f(1, 1, 1)
        glBegin(GL_POLYGON)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.25, -0.5, -0.5)
        glVertex3f(-0.25, -0.25, -0.5)
        glVertex3f(-0.5, -0.25, -0.5)
        glEnd()
        
        # Dibuja un triángulo rojo con vértices en (0.0, 0.0, 0.0), (1.0, 0.0, 0.0) y (0.5, 1.0, 0.0)
        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(0.25, 0.25, 0.25)
        glVertex3f(0.75, 0.5, 0.25)
        glVertex3f(0.5, 0.75, 0.25)
        glEnd()

        # Dibujar los vértices de las figuras y las coordenadas
        self.draw_vertices()

        glFlush()

    def draw_vertices(self):
        # Dibujar vértices del rectángulo
        glColor3f(0, 1, 0)  # Color verde para los puntos de los vértices
        glPointSize(5)
        glBegin(GL_POINTS)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.25, -0.5, -0.5)
        glVertex3f(-0.25, -0.25, -0.5)
        glVertex3f(-0.5, -0.25, -0.5)
        glEnd()

        # Dibujar vértices del triángulo
        glColor3f(0, 1, 0)  # Color verde para los puntos de los vértices
        glBegin(GL_POINTS)
        glVertex3f(0.25, 0.25, 0.25)
        glVertex3f(0.75, 0.5, 0.25)
        glVertex3f(0.5, 0.75, 0.25)
        glEnd()

        # Dibujar texto para las coordenadas de los vértices
        glColor3f(0, 1, 0)  # Color verde para el texto
        self.draw_text("(-0.5, -0.5)", -0.8, -0.5)
        self.draw_text("(-0.25, -0.5)", -0.25, -0.5)
        self.draw_text("(-0.25, -0.25)", -0.25, -0.25)
        self.draw_text("(-0.5, -0.25)", -0.8, -0.25)

        self.draw_text("(0.25, 0.25)", 0.25, 0.25)
        self.draw_text("(0.75, 0.5)", 0.75, 0.5)
        self.draw_text("(0.5, 0.75)", 0.5, 0.75)

    def draw_text(self, text, x, y):
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

if __name__ == '__main__':
    IntroduccionOpenGL().run()

