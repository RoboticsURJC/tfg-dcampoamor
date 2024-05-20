/*
* Compiler parameters:
* -framework GLUT
* -framework OpenGL
*/

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// Displays a window with a square inside.

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	glBegin(GL_POLYGON);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.5, 0.0, 0.0);
		glVertex3f(0.5, 0.5, 0.0);
		glVertex3f(0.0, 0.5, 0.0);
	glEnd();
	
	glFlush();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);	// Use single display buffer.
	glutInitWindowSize(300, 300);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Hello world");
	glutDisplayFunc(display);
	glutMainLoop();
	return 0;
}
