'''Ventana GLUT abstracta que se encarga de todo el trabajo pesado'''

from OpenGL.GL	 import *
from OpenGL.GLU  import *
from OpenGL.GLUT import *

# Hay un agujero en la implementación python-opengl de glMan2f.
# La implementación en python dice:
# glMap2f(target, u1, u2, v1, v2, points)
# Esta es una firma completamente no estándar que no permite la mayoría de los usos funky con
# strides y similares, pero ha sido así durante mucho tiempo...
# La implementación en c dice:
# glMap2f(target, u1, u2, ustride, uorder, v1, v2, vstride, vorder, points)

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
		aspect = float(self.height)/float(self.width)
		# ortho is the scaling factor for the orthogonal projection
		if self.ortho:
			if(aspect > 1):
				gluOrtho2D(-self.ortho, self.ortho, -aspect, aspect)
			else:
				gluOrtho2D(-1/aspect, 1/aspect, -self.ortho, self.ortho)
		else:
			gluPerspective(30, 1.0/aspect, 1, 20)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

	def display(self):
		
		raise NotImplementedError
	
	@staticmethod
	def run():
		'''Inicia el bucle principal.'''
		glutMainLoop()

